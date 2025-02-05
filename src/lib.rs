#![allow(non_snake_case)]

use einsum_derive::einsum;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::{QR, SVD};
use num::complex::{c64, Complex};
use pyo3::prelude::*;
use quantypes::QuantumCircuit;
use rand::prelude::*;
use std::cmp;
use std::f64::consts::PI;

mod stabilizer_simulator;
use stabilizer_simulator::{
    check_commutes, multiply_pauli_complex_phase, multiply_pauli_inplace, StabilizerSimulator,
};

#[derive(Clone, Debug)]
#[pyclass]
struct MatrixProductState {
    #[pyo3(get, set)]
    max_bond: usize,
    bond_dims: Array1<usize>,
    sites: usize,
    arrays: Vec<Array3<Complex<f64>>>,
    largest_bond: usize,
}

#[allow(unused)]
#[pymethods]
impl MatrixProductState {
    #[new]
    fn new(sites: usize, max_bond: usize) -> MatrixProductState {
        MatrixProductState {
            max_bond,
            bond_dims: Array::ones(sites - 1),
            sites,
            arrays: MatrixProductState::gen_arrays(sites),
            largest_bond: 0,
        }
    }

    fn project_magic_n(&mut self, q: usize) {
        // Do nothing -- For consistency with MAST
    }

    fn project_magic(&mut self) {
        // Do nothing -- For consistency with MAST
    }

    fn get_largest_bond(&self) -> usize {
        self.largest_bond
    }

    fn left_canonize(&mut self) {
        for j in (1..self.sites).rev() {
            self.left_canonize_site(j);
        }
    }

    fn right_canonize(&mut self) {
        for j in (0..self.sites - 1) {
            self.right_canonize_site(j);
        }
    }

    fn right_compress(&mut self) {
        for j in (0..self.sites - 1) {
            self.right_compress_site(j);
        }
    }

    fn mixed_canonize(&mut self, i: usize) {
        for j in 0..i {
            self.right_canonize_site(j);
        }
        for j in (i + 1..self.sites).rev() {
            self.left_canonize_site(j);
        }
    }

    fn compress(&mut self) {
        self.left_canonize();
        self.right_compress();
    }

    fn schmidt_values(&self, i: usize) -> Vec<f64> {
        let mut self_clone = self.clone();

        self_clone.mixed_canonize(i);

        // Next contract and do some svd
        let contracted = einsum!(
            "ikj,jlm->iklm",
            self_clone.arrays[i].view(),
            self_clone.arrays[i + 1].view(),
        );

        let shape = contracted.shape();
        let contracted = contracted
            .to_shape((shape[0] * shape[1], shape[2] * shape[3]))
            .unwrap();
        let (_, s, _) = contracted.svd(false, false).unwrap();
        let s = Array::from_iter(s.into_iter().filter(|s| s.abs() > 1e-10)).mapv(|a| a.powf(2.));

        let (v, offset) = s.into_raw_vec_and_offset();
        v
    }

    fn get_bond_dim(&self, site: usize) -> usize {
        self.bond_dims[site]
    }

    fn h(&mut self, site: usize) {
        let H = array![
            [0.5_f64.sqrt(), 0.5_f64.sqrt()],
            [0.5_f64.sqrt(), -(0.5_f64.sqrt())]
        ];
        self.apply_one_qubit_gate(H.map(|x| x.into()).view(), site);
    }

    fn z(&mut self, site: usize) {
        let Z = array![
            [Complex::new(1., 0.), Complex::new(0., 0.)],
            [Complex::new(0., 0.), Complex::new(-1., 0.)]
        ];

        self.apply_one_qubit_gate(Z.view(), site);
    }

    fn y(&mut self, site: usize) {
        let Y = array![
            [Complex::new(0., 0.), Complex::new(0., -1.)],
            [Complex::new(0., 1.), Complex::new(0., 0.)]
        ];

        self.apply_one_qubit_gate(Y.view(), site);
    }

    fn x(&mut self, site: usize) {
        let X = array![
            [Complex::new(0., 0.), Complex::new(1., 0.)],
            [Complex::new(1., 0.), Complex::new(0., 0.)]
        ];

        self.apply_one_qubit_gate(X.view(), site);
    }

    fn s(&mut self, site: usize) {
        let S = array![[c64(1., 0.), c64(0., 0.)], [c64(0., 0.), c64(0., 1.)]];

        self.apply_one_qubit_gate(S.view(), site);
    }

    fn sdg(&mut self, site: usize) {
        let Sdg = array![
            [Complex::new(1., 0.), Complex::new(0., 0.)],
            [Complex::new(0., 0.), Complex::new(0., -1.)]
        ];

        self.apply_one_qubit_gate(Sdg.view(), site);
    }

    fn crz(&mut self, angle: f64, control: usize, target: usize) {
        self.rz(angle / 2., target);
        self.cx(control, target);
        self.rz(-angle / 2., target);
        self.cx(control, target);
    }

    fn toff(&mut self, control_1: usize, control_2: usize, target: usize) {
        self.h(target);
        self.cx(control_2, target);
        self.tdg(target);
        self.cx(control_1, target);
        self.t(target);
        self.cx(control_2, target);
        self.tdg(target);
        self.cx(control_1, target);
        self.t(target);
        self.t(control_2);
        self.h(target);
        self.cx(control_1, control_2);
        self.t(control_1);
        self.tdg(control_2);
        self.cx(control_1, control_2);
    }

    fn cswap(&mut self, control: usize, a: usize, b: usize) {
        self.cx(b, a);
        self.toff(control, a, b);
        self.cx(b, a);
    }

    fn ccz(&mut self, a: usize, b: usize, c: usize) {
        self.h(c);
        self.toff(a, b, c);
        self.h(c);
    }

    fn project(&mut self, site: usize) -> i32 {
        let eval = self.z_expectation(site);

        let prob = (1. + eval) / 2.;
        let mut rng = rand::thread_rng();
        let random: f64 = rng.gen(); // generates a float between 0 and 1

        let mut outcome = if random > prob { 1 } else { 0 };

        if outcome == 1 {
            self.P_1(site);
        } else {
            self.P_0(site);
        }

        outcome
    }

    fn P_0(&mut self, site: usize) {
        let P_0 = array![
            [Complex::new(1., 0.), Complex::new(0., 0.)],
            [Complex::new(0., 0.), Complex::new(0., 0.)]
        ];

        self.apply_one_qubit_gate(P_0.view(), site);
        self.renormalize();
        self.compress();
    }

    fn P_1(&mut self, site: usize) {
        let P_1 = array![
            [Complex::new(0., 0.), Complex::new(0., 0.)],
            [Complex::new(0., 0.), Complex::new(1., 0.)]
        ];

        self.apply_one_qubit_gate(P_1.view(), site);
        self.renormalize();
        self.compress();
    }

    fn iden(&mut self, ctrl: usize, targ: usize) {
        let iden = array![
            [1.0_f64, 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ];
        self.apply_two_qubit_gate(iden.map(|x| x.into()).view(), ctrl, targ);
    }

    fn cx(&mut self, ctrl: usize, targ: usize) {
        if targ < ctrl {
            return self.xc(targ, ctrl);
        }

        let CX = array![
            [1.0_f64, 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 1., 0.]
        ];
        self.apply_two_qubit_gate(CX.map(|x| x.into()).view(), ctrl, targ);
    }

    fn cnot(&mut self, ctrl: usize, targ: usize) {
        self.cx(ctrl, targ)
    }

    fn cz(&mut self, ctrl: usize, targ: usize) {
        self.h(targ);
        self.cx(ctrl, targ);
        self.h(targ)
    }

    fn t(&mut self, site: usize) {
        let T = array![
            [1.0.into(), 0.0.into()],
            [0.0.into(), Complex::from_polar(1., PI / 4.)]
        ];

        self.apply_one_qubit_gate(T.view(), site);
    }

    fn tdg(&mut self, site: usize) {
        let Tdg = array![
            [1.0.into(), 0.0.into()],
            [0.0.into(), Complex::from_polar(1., -PI / 4.)]
        ];

        self.apply_one_qubit_gate(Tdg.view(), site);
    }

    fn xc(&mut self, targ: usize, ctrl: usize) {
        let XC = array![
            [1.0_f64, 0., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.]
        ];
        self.apply_two_qubit_gate(XC.map(|x| x.into()).view(), targ, ctrl);
    }

    fn rz(&mut self, angle: f64, site: usize) {
        self.h(site);
        self.rx(site, angle);
        self.h(site);
    }

    fn swap(&mut self, dest: usize, orig: usize) {
        self.swap_site_to(dest, orig);
    }

    fn rx(&mut self, site: usize, angle: f64) {
        let RX = array![
            [
                c64((angle / 2.).cos(), 0.),
                c64(0., -1. * (angle / 2.).sin())
            ],
            [
                c64(0., -1. * (angle / 2.).sin()),
                c64((angle / 2.).cos(), 0.)
            ]
        ];
        self.apply_one_qubit_gate(RX.view(), site);
    }

    fn z_expectation(&self, site: usize) -> f64 {
        let Z = array![
            [Complex::new(1., 0.), Complex::new(0., 0.)],
            [Complex::new(0., 0.), Complex::new(-1., 0.)]
        ];
        self.local_expectation(Z.view(), site)
    }

    fn expectation(&self, mut zs: Vec<i8>, mut xs: Vec<i8>) -> f64 {
        assert!(zs.len() == self.sites, "zs must match the number of qubits");
        assert!(xs.len() == self.sites, "xs must match the number of qubits");

        let mut tn_copy = self.clone();
        let mut min = 0;
        let mut max = self.sites;

        tn_copy.mixed_canonize(min);
        let tn_copy_h = tn_copy.clone();

        for (i, j) in zs.iter().enumerate() {
            if *j == 1 {
                tn_copy.z(i);
            }
        }

        for (i, j) in xs.iter().enumerate() {
            if *j == 1 {
                tn_copy.x(i);
            }
        }

        let bra = tn_copy_h.arrays[min]
            .view()
            .reversed_axes()
            .map(|x| x.conj());
        let ket = tn_copy.arrays[min].clone();
        let mut expec = einsum!("ijk,mji->km", ket, bra);

        for site in min + 1..max {
            let bra = tn_copy_h.arrays[site]
                .view()
                .reversed_axes()
                .map(|x| x.conj());
            let ket = tn_copy.arrays[site].clone();
            expec = einsum!("km,kbc,dbm->cd", expec, ket, bra);
        }
        let eval = expec.sum();

        approx::assert_abs_diff_eq!(0., eval.im, epsilon = 1e-10);
        eval.re
    }

    fn get_sv(&self) -> Vec<Complex<f64>> {
        let sv_0 = self.arrays.last().unwrap().clone();
        self.arrays
            .iter()
            .rev()
            .skip(1)
            .fold(sv_0, |sv, array| {
                let sv_ = einsum!("ijk,kml->ijml", array.view(), sv.view());
                let shape = sv_.shape().to_vec();
                sv_.into_shape_with_order((shape[0], shape[1] * shape[2], shape[3]))
                    .unwrap()
            })
            .into_shape_with_order(2_u32.pow(self.sites as u32) as usize)
            .unwrap()
            .to_vec()
    }

    fn apply_mps(&self, other: MatrixProductState) -> f64 {
        if self.sites != other.sites {
            panic!("Cannot apply MPS to MPS of different length");
        }

        let mut self_clone = self.clone();
        let other_clone = other.clone();

        let ket = self_clone.arrays[0].clone();
        let bra = other_clone.arrays[0]
            .view()
            .reversed_axes()
            .map(|x| x.conj());
        let mut bra_ket_prev = einsum!("ijk,ljm->ikml", ket.view(), bra);

        for n in 1..(self_clone.sites) {
            let ket = self_clone.arrays[n].clone();
            let bra = other_clone.arrays[n]
                .view()
                .reversed_axes()
                .map(|x| x.conj());
            let bra_ket = einsum!("ijk,ljm->ikml", ket.view(), bra);
            bra_ket_prev = einsum!("ikml,kbld->ibmd", bra_ket_prev, bra_ket)
        }
        bra_ket_prev.sum().re
    }

    fn add_qubit(&mut self) {
        self.sites += 1;
        self.arrays.append(&mut MatrixProductState::gen_arrays(1));
        self.bond_dims.append(Axis(0), array![1].view());
    }
}

#[allow(unused)]
impl MatrixProductState {
    fn gen_arrays(sites: usize) -> Vec<Array3<Complex<f64>>> {
        let mut vec = Vec::new();
        for n in 0..sites {
            vec.push(
                Array3::<Complex<f64>>::from_shape_vec([1, 2, 1], vec![1.0.into(), 0.0.into()])
                    .unwrap(),
            );
        }
        vec
    }

    fn swap_site_to(&mut self, dest: usize, orig: usize) {
        let SWAP = array![
            [1.0_f64, 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.]
        ];

        if dest < orig {
            for k in (dest..orig).rev() {
                self.apply_two_qubit_gate(SWAP.map(|x| x.into()).view(), k, k + 1);
            }
        }
        if orig < dest {
            for k in orig..dest {
                self.apply_two_qubit_gate(SWAP.map(|x| x.into()).view(), k, k + 1);
            }
        }
    }

    fn apply_two_qubit_gate(&mut self, op: ArrayView2<Complex<f64>>, ctrl: usize, targ: usize) {
        if targ != ctrl + 1 {
            self.swap_site_to(ctrl + 1, targ);
        }
        let mut renorm = false;
        let tensor_op = op.into_shape_with_order((2, 2, 2, 2)).unwrap();
        let left_bond = self.arrays[ctrl].shape()[0];
        let right_bond = self.arrays[ctrl + 1].shape()[1];
        let contracted = einsum!(
            "ikj,jlm,nokl->inom",
            self.arrays[ctrl].view(),
            self.arrays[ctrl + 1].view(),
            tensor_op,
        );
        let shape = contracted.shape();
        let contracted = contracted
            .to_shape((shape[0] * shape[1], shape[2] * shape[3]))
            .unwrap();
        let (u, s, v) = contracted.svd(true, true).unwrap();
        if self.max_bond > 0 && s.shape()[0] > self.max_bond {
            renorm = true;
        }
        let mut s = Array::from_iter(s.into_iter().filter(|s| s.abs() > 1e-10));
        if self.max_bond > 0 && s.shape()[0] > self.max_bond {
            s = s.slice(s![..self.max_bond]).to_owned();
        }

        let u = u.unwrap().slice(s![.., ..s.shape()[0]]).to_owned();
        let v = v.unwrap().slice(s![..s.shape()[0], ..]).to_owned();
        let s = Array2::from_diag(&s.map(|x| x.into()));
        self.bond_dims[ctrl] = s.shape()[0];
        if s.shape()[0] > self.largest_bond {
            self.largest_bond = s.shape()[0];
        }
        let sv = s.dot(&v);
        self.arrays[ctrl] = u
            .into_shape_with_order((shape[0], shape[1], s.shape()[0]))
            .unwrap();
        self.arrays[ctrl + 1] = sv
            .into_shape_with_order((s.shape()[0], shape[2], shape[3]))
            .unwrap();
        if renorm {
            self.renormalize();
        }
        if targ != ctrl + 1 {
            self.swap_site_to(targ, ctrl + 1);
        }
    }

    fn right_canonize_site(&mut self, i: usize) {
        if i == self.sites - 1 {
            panic!("Cannot canonize {} and {} sites", i, i + 1)
        }

        let contracted = einsum!(
            "ikj,jlm->iklm",
            self.arrays[i].view(),
            self.arrays[i + 1].view(),
        );

        let shape = contracted.shape();
        let contracted = contracted
            .to_shape((shape[0] * shape[1], shape[2] * shape[3]))
            .unwrap();
        let (q, r) = contracted.qr().unwrap();

        self.bond_dims[i] = *q.shape().last().unwrap();

        self.arrays[i] = q
            .clone()
            .into_shape_with_order((shape[0], shape[1], *q.shape().last().unwrap()))
            .unwrap();
        self.arrays[i + 1] = r
            .clone()
            .into_shape_with_order((*r.shape().first().unwrap(), shape[2], shape[3]))
            .unwrap();
    }

    fn right_compress_site(&mut self, i: usize) {
        if i == self.sites - 1 {
            panic!("Cannot canonize {} and {} sites", i, i + 1)
        }

        let contracted = einsum!(
            "ikj,jlm->iklm",
            self.arrays[i].view(),
            self.arrays[i + 1].view(),
        );

        let shape = contracted.shape();
        let contracted = contracted
            .to_shape((shape[0] * shape[1], shape[2] * shape[3]))
            .unwrap();
        let (u, s, v) = contracted.svd(true, true).unwrap();
        let mut s = Array::from_iter(s.into_iter().filter(|s| s.abs() > 1e-10));
        if self.max_bond > 0 && s.shape()[0] > self.max_bond {
            s = s.slice(s![..self.max_bond]).to_owned();
        }

        let u = u.unwrap().slice(s![.., ..s.shape()[0]]).to_owned();
        let v = v.unwrap().slice(s![..s.shape()[0], ..]).to_owned();
        let s = Array2::from_diag(&s.map(|x| x.into()));
        self.bond_dims[i] = s.shape()[0];
        let sv = s.dot(&v);
        self.arrays[i] = u
            .into_shape_with_order((shape[0], shape[1], s.shape()[0]))
            .unwrap();
        self.arrays[i + 1] = sv
            .into_shape_with_order((s.shape()[0], shape[2], shape[3]))
            .unwrap();
    }

    fn left_canonize_site(&mut self, i: usize) {
        if i == 0 {
            panic!("Cannot canonize {} and {} sites", i - 1, i)
        }

        let contracted = einsum!(
            "ijk,klm->mlji",
            self.arrays[i - 1].view(),
            self.arrays[i].view(),
        );

        let shape = contracted.shape();
        let contracted = contracted
            .to_shape((shape[0] * shape[1], shape[2] * shape[3]))
            .unwrap();
        let (q, r) = contracted.qr().unwrap();

        self.bond_dims[i - 1] = *q.shape().last().unwrap();

        let q = q
            .clone()
            .into_shape_with_order((shape[0], shape[1], *q.shape().last().unwrap()))
            .unwrap();
        let r = r
            .clone()
            .into_shape_with_order((*r.shape().first().unwrap(), shape[2], shape[3]))
            .unwrap();
        let q = einsum!("mla->alm", q);
        let r = einsum!("aji->ija", r);

        self.arrays[i - 1] = r;
        self.arrays[i] = q;
    }

    fn renormalize(&mut self) {
        let norm = self.local_expectation(Array2::<Complex<f64>>::eye(2).view(), 0);
        self.arrays[0] = self.arrays[0].clone() / norm.sqrt();
    }

    fn apply_one_qubit_gate(&mut self, op: ArrayView2<Complex<f64>>, site: usize) {
        self.arrays[site] = einsum!("ikj,lk->ilj", self.arrays[site].view(), op.view());
    }

    fn local_expectation(&self, op: ArrayView2<Complex<f64>>, site: usize) -> f64 {
        // First put the MPS into a mixed canonical form at the site
        let mut self_clone = self.clone();
        self_clone.mixed_canonize(site);
        if site >= self_clone.sites {
            panic!("Site is greater than the system size!");
        }
        let bra = self_clone.arrays[site]
            .view()
            .reversed_axes()
            .map(|x| x.conj());
        self_clone.apply_one_qubit_gate(op.view(), site);
        let ket = self_clone.arrays[site].clone();
        let expec = einsum!("ijk,kji", ket, bra);
        expec.sum().re
    }
}

#[allow(unused)]
#[pyclass]
#[derive(Clone, Debug)]
struct StabilizerTensorNetwork {
    #[pyo3(get, set)]
    tensor_network: MatrixProductState,
    stabilizer_simulator: StabilizerSimulator,
    #[pyo3(get, set)]
    num_qubits: usize,
}

#[pymethods]
impl StabilizerTensorNetwork {
    #[new]
    fn new(n: usize, max_bond: usize) -> StabilizerTensorNetwork {
        StabilizerTensorNetwork {
            tensor_network: MatrixProductState::new(n, max_bond),
            stabilizer_simulator: StabilizerSimulator::new(n),
            num_qubits: n,
        }
    }

    fn project_magic_n(&mut self, _q: usize) {
        // Do nothing -- For consistency with MAST
    }

    fn project_magic(&mut self) {
        // Do nothing -- For consistency with MAST
    }

    fn add_qubit(&mut self) {
        self.tensor_network.add_qubit();
        self.stabilizer_simulator.add_qubit();
        self.num_qubits += 1;
    }

    fn schmidt_values(&self, i: usize) -> Vec<f64> {
        self.tensor_network.schmidt_values(i)
    }

    fn get_bond_dim(&self, site: usize) -> usize {
        self.tensor_network.get_bond_dim(site)
    }

    fn get_largest_bond(&self) -> usize {
        self.tensor_network.get_largest_bond()
    }

    fn h(&mut self, q: usize) {
        self.stabilizer_simulator.h(q);
    }

    fn x(&mut self, q: usize) {
        self.stabilizer_simulator.x(q);
    }

    fn y(&mut self, q: usize) {
        self.stabilizer_simulator.y(q);
    }

    fn z(&mut self, q: usize) {
        self.stabilizer_simulator.z(q);
    }

    fn s(&mut self, q: usize) {
        self.stabilizer_simulator.s(q);
    }

    fn sdg(&mut self, q: usize) {
        self.stabilizer_simulator.s(q);
        self.stabilizer_simulator.z(q);
    }

    fn cx(&mut self, control: usize, target: usize) {
        self.cnot(control, target);
    }

    fn cnot(&mut self, control: usize, target: usize) {
        self.stabilizer_simulator.cnot(control, target);
    }

    fn cz(&mut self, a: usize, b: usize) {
        self.h(b);
        self.cx(a, b);
        self.h(b);
    }

    fn t(&mut self, q: usize) {
        let (gate_coeffs, destab_list, stab_list) = self.rz_decomp(PI / 4., q);
        self.update_tn(gate_coeffs, stab_list, destab_list);
    }

    fn tdg(&mut self, q: usize) {
        let (gate_coeffs, destab_list, stab_list) = self.rz_decomp(-PI / 4., q);
        self.update_tn(gate_coeffs, stab_list, destab_list);
    }

    fn rz(&mut self, angle: f64, q: usize) {
        let (gate_coeffs, destab_list, stab_list) = self.rz_decomp(angle, q);
        self.update_tn(gate_coeffs, stab_list, destab_list);
    }

    fn crz(&mut self, angle: f64, control: usize, target: usize) {
        self.rz(angle / 2., target);
        self.cx(control, target);
        self.rz(-angle / 2., target);
        self.cx(control, target);
    }

    fn toff(&mut self, control_1: usize, control_2: usize, target: usize) {
        self.h(target);
        self.ccz(control_1, control_2, target);
        self.h(target);
    }

    fn toff_t(&mut self, control_1: usize, control_2: usize, target: usize) {
        self.h(target);
        self.cx(control_2, target);
        self.tdg(target);
        self.cx(control_1, target);
        self.t(target);
        self.cx(control_2, target);
        self.tdg(target);
        self.cx(control_1, target);
        self.t(target);
        self.t(control_2);
        self.h(target);
        self.cx(control_1, control_2);
        self.t(control_1);
        self.tdg(control_2);
        self.cx(control_1, control_2);
    }

    fn ccz(&mut self, a: usize, b: usize, c: usize) {
        let (gate_coeffs, destab_list, stab_list) = self.ccz_decomp(array![a, b, c]);
        self.update_tn(gate_coeffs, stab_list, destab_list);
    }

    fn ccz_t(&mut self, a: usize, b: usize, c: usize) {
        self.h(c);
        self.toff_t(a, b, c);
        self.h(c);
    }

    fn cswap(&mut self, control: usize, a: usize, b: usize) {
        self.cx(b, a);
        self.toff(control, a, b);
        self.cx(b, a);
    }

    fn expectation(&self, mut zs: Vec<i8>, mut xs: Vec<i8>) -> f64 {
        assert!(
            zs.len() == self.num_qubits,
            "zs must match the number of qubits"
        );
        assert!(
            xs.len() == self.num_qubits,
            "xs must match the number of qubits"
        );
        zs.append(&mut xs);
        let gate: Array1<_> = zs.into();
        let (coeff, destab, stab) = self.gate_decomposition(gate.view());
        let eval = self.evaluate_tn(coeff, stab, destab);
        approx::assert_abs_diff_eq!(0., eval.im, epsilon = 1e-10);
        eval.re
    }

    fn project(&mut self, q: usize) -> i32 {
        self._project(q, false)
    }

    // Project qubit q onto the computational basis
    fn _project(&mut self, q: usize, magic: bool) -> i32 {
        let mut xs = vec![0_i8; self.num_qubits];
        let mut zs = vec![0_i8; self.num_qubits];
        zs[q] = 1;
        zs.append(&mut xs);

        let mut gate: Array1<_> = zs.into();
        let (coeff, destab, stab) = self.gate_decomposition(gate.view());
        let eval = self.evaluate_tn(coeff, stab.clone(), destab.clone());
        approx::assert_abs_diff_eq!(0., eval.im, epsilon = 1e-10);

        let prob = (1. + eval.re) / 2.;
        let mut rng = rand::thread_rng();
        let random: f64 = rng.gen(); // generates a float between 0 and 1

        let mut outcome = if random > prob { 1 } else { 0 };
        if magic {
            approx::assert_abs_diff_eq!(0., eval.re, epsilon = 1e-10);
            outcome = 0;
        }

        let factor = coeff * (-1.0_f64).powi(outcome);

        // Build the vector of operators to perform. We do it this way so that there
        // is a greater chance of non-identity operators being next to each other
        let mut operators = Vec::new();
        let mut Ys = 0.;
        let mut theta_qubit = None;
        for (i, (d, s)) in destab.iter().zip(stab.iter()).enumerate() {
            if *s {
                if *d {
                    operators.push("Y");
                    theta_qubit = Some(i);
                    Ys += 1.;
                } else {
                    operators.push("Z");
                    theta_qubit = Some(i);
                }
            } else if *d {
                operators.push("X");
                theta_qubit = Some(i);
            } else {
                operators.push("I");
            }
        }

        let mut q1 = None;
        for (q2, &op) in operators.iter().enumerate() {
            if op == "Z" {
                self.tensor_network.h(q2);
            } else if op == "Y" {
                self.tensor_network.s(q2);
            } else if op == "I" {
                continue;
            }
            if q1.is_none() {
                q1 = Some(q2);
                continue;
            }
            self.tensor_network.cx(q2, q1.unwrap());
            q1 = Some(q2);
        }
        let renorm = (1. + eval.re.abs()).sqrt();

        let rot_matrix = array![
            [
                Complex::new(0.5_f64.sqrt() * renorm, 0.),
                Complex::new(0.0, 1.0_f64).powf(Ys) * factor * 0.5_f64.sqrt() * renorm
            ],
            [
                Complex::new(0.0, 1.0_f64).powf(Ys) * factor * 0.5_f64.sqrt() * renorm,
                Complex::new(0.5_f64.sqrt() * renorm, 0.)
            ]
        ];
        self.tensor_network
            .apply_one_qubit_gate(rot_matrix.view(), theta_qubit.unwrap());
        q1 = None;
        for (q2, &op) in operators.iter().enumerate().rev() {
            if op == "I" {
                continue;
            }
            if q1.is_none() {
                q1 = Some(q2);
                continue;
            }
            self.tensor_network.cx(q1.unwrap(), q2);
            q1 = Some(q2);
        }
        for (q2, &op) in operators.iter().enumerate() {
            if op == "Z" {
                self.tensor_network.h(q2);
            } else if op == "Y" {
                self.tensor_network.sdg(q2);
            }
        }

        let fdt = destab.iter().position(|x| *x);
        gate.append(Axis(0), array![outcome as i8].view()).unwrap();
        if let Some(k) = fdt {
            let destab_update = array![
                [Complex::new(0.5_f64.sqrt(), 0.), Complex::new(0., 0.)],
                [Complex::new(0., 0.), Complex::new(0., 0.)]
            ];
            self.tensor_network
                .apply_one_qubit_gate(destab_update.view(), k);

            let stab_row = self
                .stabilizer_simulator
                .storage
                .slice(self.stabilizer_simulator.stab_row(k))
                .to_owned();
            for (i, s) in stab.iter().enumerate() {
                if *s && i != k {
                    self.stabilizer_simulator
                        .multiply_rows(self.num_qubits + i, k);
                }
            }

            for (i, d) in destab.iter().enumerate() {
                if *d && i != k {
                    self.stabilizer_simulator.multiply_rows(i, k);
                }
            }
            self.stabilizer_simulator
                .storage
                .slice_mut(self.stabilizer_simulator.destab_row(k))
                .assign(&stab_row);
            self.stabilizer_simulator
                .storage
                .slice_mut(self.stabilizer_simulator.stab_row(k))
                .assign(&gate);
        }

        // Finally renormalize
        self.tensor_network.compress();
        self.tensor_network.renormalize();
        outcome
    }
}

impl StabilizerTensorNetwork {
    fn evaluate_tn(
        &self,
        coeff: Complex<f64>,
        stabs: Array1<bool>,
        destabs: Array1<bool>,
    ) -> Complex<f64> {
        let mut tn_copy = self.tensor_network.clone();
        let mut min = 0;
        let mut max = self.num_qubits;

        // First we apply an X where the destab is in the decomp
        for (i, j) in stabs.indexed_iter() {
            if *j {
                min = cmp::min(min, i);
                max = cmp::max(max, i);
            }
        }

        // Next we apply an Z where the stab is in the decomp
        for (i, j) in destabs.indexed_iter() {
            if *j {
                min = cmp::min(min, i);
                max = cmp::max(max, i);
            }
        }
        min = 0;
        max = self.num_qubits;

        tn_copy.mixed_canonize(min);
        let tn_copy_h = tn_copy.clone();

        // First we apply an X where the destab is in the decomp
        for (i, j) in stabs.indexed_iter() {
            if *j {
                tn_copy.z(i);
            }
        }

        // Next we apply an Z where the stab is in the decomp
        for (i, j) in destabs.indexed_iter() {
            if *j {
                tn_copy.x(i);
            }
        }

        let bra = tn_copy_h.arrays[min]
            .view()
            .reversed_axes()
            .map(|x| x.conj());
        let ket = tn_copy.arrays[min].clone();
        let mut expec = einsum!("ijk,mji->km", ket, bra);

        for site in min + 1..max {
            let bra = tn_copy_h.arrays[site]
                .view()
                .reversed_axes()
                .map(|x| x.conj());
            let ket = tn_copy.arrays[site].clone();
            expec = einsum!("km,kbc,dbm->cd", expec, ket, bra);
        }
        coeff * expec.sum()
    }

    fn update_tn(
        &mut self,
        coeffs: Array1<Complex<f64>>,
        stabs: Array2<bool>,
        destabs: Array2<bool>,
    ) {
        // First we apply an X where the destab is in the decomp
        for (i, j) in stabs.slice(s![0, ..]).indexed_iter() {
            if *j {
                self.tensor_network.z(i);
            }
        }

        // Next we apply an Z where the stab is in the decomp
        for (i, j) in destabs.slice(s![0, ..]).indexed_iter() {
            if *j {
                self.tensor_network.x(i);
            }
        }

        if coeffs.shape()[0] > 1 {
            let X_IxIy = Zip::from(destabs.slice(s![0, ..]))
                .and(destabs.slice(s![1, ..]))
                .map_collect(|&a, &b| a ^ b);
            let Z_IyIz = Zip::from(stabs.slice(s![0, ..]))
                .and(stabs.slice(s![1, ..]))
                .map_collect(|&a, &b| a ^ b);

            // Build the vector of operators to perform. We do it this way so that there
            // is a greater chance of non-identity operators being next to each other
            let mut operators = Vec::new();
            let mut theta_qubit = None;
            let mut Ys = 0;
            for i in 0..X_IxIy.len() {
                if X_IxIy[i] && Z_IyIz[i] {
                    operators.push("Y");
                    Ys += 1;
                    theta_qubit = Some(i);
                } else if X_IxIy[i] {
                    operators.push("X");
                    theta_qubit = Some(i);
                } else if Z_IyIz[i] {
                    operators.push("Z");
                    theta_qubit = Some(i);
                } else {
                    operators.push("I");
                }
            }

            let theta = (coeffs[0].powf(2.) + coeffs[1].norm_sqr()).sqrt();
            let (_, _phi) = coeffs[1].to_polar();

            let prefactor = Complex::new(0., 1.) * Complex::new(0., -1.).powf(Ys.into()).conj();
            let theta = (prefactor * coeffs[1] / theta).re.asin();

            // Apply CRX
            let mut q1 = None;
            for (q2, &op) in operators.iter().enumerate() {
                if op == "Z" {
                    self.tensor_network.h(q2);
                } else if op == "Y" {
                    self.tensor_network.s(q2);
                } else if op == "I" {
                    continue;
                }
                if q1.is_none() {
                    q1 = Some(q2);
                    continue;
                }
                self.tensor_network.cx(q2, q1.unwrap());
                q1 = Some(q2);
            }
            self.tensor_network.rx(theta_qubit.unwrap(), 2. * theta);
            q1 = None;
            for (q2, &op) in operators.iter().enumerate().rev() {
                if op == "I" {
                    continue;
                }
                if q1.is_none() {
                    q1 = Some(q2);
                    continue;
                }
                self.tensor_network.cx(q1.unwrap(), q2);
                q1 = Some(q2);
            }
            for (q2, &op) in operators.iter().enumerate() {
                if op == "Z" {
                    self.tensor_network.h(q2);
                } else if op == "Y" {
                    self.tensor_network.sdg(q2);
                }
            }
        }
    }

    fn gate_decomposition(
        &self,
        gate: ArrayView1<i8>,
    ) -> (Complex<f64>, Array1<bool>, Array1<bool>) {
        if gate == Array::zeros(gate.len()) {
            return (
                1.0.into(),
                Array1::from_elem(self.num_qubits, false),
                Array1::from_elem(self.num_qubits, false),
            );
        }

        let stab_v = self
            .stabilizer_simulator
            .storage
            .slice(s![self.num_qubits..2 * self.num_qubits, ..])
            .map_axis(Axis(1), |s| !check_commutes(gate.view(), s));
        let destab_v = self
            .stabilizer_simulator
            .storage
            .slice(s![..self.num_qubits, ..])
            .map_axis(Axis(1), |d| !check_commutes(gate.view(), d));

        // calculated the phase
        let (c_phase, row_phase) = destab_v.iter().enumerate().filter(|(_, &x)| x).rev().fold(
            (Complex::<f64>::ONE, Array1::zeros(2 * self.num_qubits + 1)),
            |(c_phase, mut row_phase), (i, _)| {
                let destab_row = self
                    .stabilizer_simulator
                    .storage
                    .slice(self.stabilizer_simulator.destab_row(i));
                let c_phase = c_phase * multiply_pauli_complex_phase(row_phase.view(), destab_row);
                multiply_pauli_inplace(&mut row_phase.view_mut(), destab_row);
                (c_phase, row_phase)
            },
        );

        // then with the stabilizers...
        let (c_phase, _) = stab_v.iter().enumerate().filter(|(_, &x)| x).rev().fold(
            (c_phase, row_phase),
            |(c_phase, mut row_phase), (i, _)| {
                let stab_row = self
                    .stabilizer_simulator
                    .storage
                    .slice(self.stabilizer_simulator.stab_row(i));
                let c_phase = c_phase * multiply_pauli_complex_phase(row_phase.view(), stab_row);
                multiply_pauli_inplace(&mut row_phase.view_mut(), stab_row);
                (c_phase, row_phase)
            },
        );

        (c_phase, destab_v, stab_v)
    }

    /// Returns an array of complex decomposition coefficients, and two arrays of (destabilizer,
    /// stabilizer) components in the decomposition
    fn rz_decomp(
        &self,
        theta: f64,
        q: usize,
    ) -> (Array1<Complex<f64>>, Array2<bool>, Array2<bool>) {
        // decompose T gate into aI + bZ
        // decompose T gate into aI + bZ
        // [Z_0, X_0], [Z_1, X_1]
        let gate_list = array![[0, 0], [1, 0]];
        let gate_coeffs = array![c64((theta / 2.).cos(), 0.), c64(0., -(theta / 2.).sin())];

        self.gate_decomp_from_list(array![q], gate_list, gate_coeffs)
    }

    /// Returns an array of complex decomposition coefficients, and two arrays of (destabilizer,
    /// stabilizer) components in the decomposition
    fn ccz_decomp(&self, qs: Array1<usize>) -> (Array1<Complex<f64>>, Array2<bool>, Array2<bool>) {
        // decompose T gate into aI + bZ
        // decompose T gate into aI + bZ
        // [Z_0, X_0], [Z_1, X_1]
        let gate_list = array![
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ];
        let gate_coeffs = array![
            3. / 4.,
            1. / 4.,
            1. / 4.,
            1. / 4.,
            -1. / 4.,
            -1. / 4.,
            -1. / 4.,
            1. / 4.
        ]
        .map(|x| x.into());

        self.gate_decomp_from_list(qs, gate_list, gate_coeffs)
    }

    fn gate_decomp_from_list(
        &self,
        qs: Array1<usize>,
        gate_list: Array2<i8>,
        mut gate_coeffs: Array1<Complex<f64>>,
    ) -> (Array1<Complex<f64>>, Array2<bool>, Array2<bool>) {
        let mut destab_list = Array::from_elem((gate_coeffs.len(), self.num_qubits), false);
        let mut stab_list = Array::from_elem((gate_coeffs.len(), self.num_qubits), false);

        for (i, (gate, coef)) in gate_list
            .rows()
            .into_iter()
            .zip(gate_coeffs.iter_mut())
            .enumerate()
        {
            let mut full_gate = Array1::zeros(2 * self.num_qubits + 1);
            for (j, q) in qs.iter().enumerate() {
                full_gate[*q] = gate[j];
                full_gate[self.num_qubits + *q] = gate[gate.len() / 2 + j];
            }
            let (phase, destab, stab) = self.gate_decomposition(full_gate.view());
            *coef *= phase;
            destab_list.row_mut(i).assign(&destab);
            stab_list.row_mut(i).assign(&stab);
        }

        (gate_coeffs, destab_list, stab_list)
    }
}

#[allow(unused)]
#[pyclass]
#[derive(Clone, Debug)]
struct MagicInjectedStn {
    #[pyo3(get, set)]
    inner: StabilizerTensorNetwork,
    num_qubits: usize,
    #[pyo3(get, set)]
    magic_id: usize,
    proj_from: usize,
    num_t: usize,
}

#[pymethods]
impl MagicInjectedStn {
    #[new]
    fn new(n: usize, num_t: usize, max_bond: usize) -> MagicInjectedStn {
        MagicInjectedStn {
            inner: StabilizerTensorNetwork::new(n + num_t, max_bond),
            num_qubits: n,
            magic_id: n,
            proj_from: n,
            num_t: num_t,
        }
    }

    fn schmidt_values(&self, i: usize) -> Vec<f64> {
        self.inner.tensor_network.schmidt_values(i)
    }

    fn get_bond_dim(&self, site: usize) -> usize {
        self.inner.tensor_network.get_bond_dim(site)
    }

    fn get_largest_bond(&self) -> usize {
        self.inner.tensor_network.get_largest_bond()
    }

    fn add_qubit(&mut self) {
        self.inner.add_qubit();
        self.num_qubits += 1;
    }

    fn h(&mut self, q: usize) {
        self.inner.h(q);
    }

    fn x(&mut self, q: usize) {
        self.inner.x(q);
    }

    fn y(&mut self, q: usize) {
        self.inner.y(q);
    }

    fn z(&mut self, q: usize) {
        self.inner.z(q);
    }

    fn s(&mut self, q: usize) {
        self.inner.s(q);
    }

    fn cx(&mut self, control: usize, target: usize) {
        self.cnot(control, target);
    }

    fn cnot(&mut self, control: usize, target: usize) {
        self.inner.cnot(control, target);
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.cnot(a, b);
        self.cnot(b, a);
        self.cnot(a, b);
    }

    fn cz(&mut self, a: usize, b: usize) {
        self.h(b);
        self.cnot(a, b);
        self.h(b);
    }

    fn t(&mut self, q: usize) {
        if self.magic_id > self.num_qubits + self.num_t {
            panic!("Insufficient number of non-Clifford gates specified during initialisation");
        }

        // prepare magic state
        self.h(self.magic_id);
        self.inner.t(self.magic_id);

        // inject
        self.cnot(q, self.magic_id);

        // log for later
        self.magic_id += 1;
    }

    fn tdg(&mut self, q: usize) {
        if self.magic_id > self.num_qubits + self.num_t {
            panic!("Insufficient number of non-Clifford gates specified during initialisation");
        }

        // prepare magic state
        self.h(self.magic_id);
        self.inner.tdg(self.magic_id);

        // inject
        self.cnot(q, self.magic_id);

        // log for later
        self.magic_id += 1;
    }

    fn rz(&mut self, angle: f64, q: usize) {
        if self.magic_id >= self.num_qubits + self.num_t {
            panic!("Insufficient number of non-Clifford gates specified during initialisation");
        }

        // prepare magic state
        self.h(self.magic_id);
        self.inner.rz(angle, self.magic_id);

        // inject
        self.cnot(q, self.magic_id);

        // log for later
        self.magic_id += 1;
    }

    fn crz(&mut self, angle: f64, control: usize, target: usize) {
        self.rz(angle / 2., target);
        self.cx(control, target);
        self.rz(-angle / 2., target);
        self.cx(control, target);
    }

    fn toff(&mut self, control_1: usize, control_2: usize, target: usize) {
        self.h(target);
        self.ccz(control_1, control_2, target);
        self.h(target);
    }

    fn toff_t(&mut self, control_1: usize, control_2: usize, target: usize) {
        self.h(target);
        self.cx(control_2, target);
        self.tdg(target);
        self.cx(control_1, target);
        self.t(target);
        self.cx(control_2, target);
        self.tdg(target);
        self.cx(control_1, target);
        self.t(target);
        self.t(control_2);
        self.h(target);
        self.cx(control_1, control_2);
        self.t(control_1);
        self.tdg(control_2);
        self.cx(control_1, control_2);
    }

    fn ccz(&mut self, a: usize, b: usize, c: usize) {
        if self.magic_id > self.num_qubits + self.num_t {
            panic!("Insufficient number of non-Clifford gates specified during initialisation");
        }

        // prepare magic state
        for i in self.magic_id..self.magic_id + 3 {
            self.inner.h(i);
        }
        self.inner
            .ccz(self.magic_id, self.magic_id + 1, self.magic_id + 2);

        // inject
        self.inner.cnot(a, self.magic_id);
        self.inner.cnot(b, self.magic_id + 1);
        self.inner.cnot(c, self.magic_id + 2);

        // log for later
        self.magic_id += 3;
    }

    fn ccz_t(&mut self, a: usize, b: usize, c: usize) {
        self.h(c);
        self.toff_t(a, b, c);
        self.h(c);
    }

    fn cswap(&mut self, control: usize, a: usize, b: usize) {
        self.cx(b, a);
        self.toff(control, a, b);
        self.cx(b, a);
    }

    fn project(&mut self, q: usize) -> i32 {
        self.inner.project(q)
    }

    fn project_magic_n(&mut self, q: usize) {
        self.inner._project(self.num_qubits + q, true);
    }

    fn project_magic(&mut self) {
        for i in self.proj_from..self.magic_id {
            self.inner._project(i, true);
        }
        self.proj_from = self.magic_id;
    }

    fn expectation(&self, mut zs: Vec<i8>, mut xs: Vec<i8>) -> f64 {
        for _ in self.num_qubits..self.inner.num_qubits {
            zs.push(0);
            xs.push(0);
        }
        self.inner.expectation(zs, xs)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn mast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MagicInjectedStn>()?;
    m.add_class::<StabilizerTensorNetwork>()?;
    m.add_class::<MatrixProductState>()?;
    m.add_class::<QuantumCircuit>()?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::stabilizer_simulator::row_to_matrix;
    use approx::*;

    impl StabilizerTensorNetwork {
        fn calculate_rz(&self, theta: f64, q: usize) -> Array2<Complex<f64>> {
            println!("{:?}", self.rz_decomp(theta, q));
            // get the t gate decomp
            let (gate_coeffs, destab_list, stab_list) = self.rz_decomp(theta, q);
            self.calculate_matrix_from_decomp(gate_coeffs, destab_list, stab_list)
                / (c64((theta / 2.).cos(), 0.) + c64(0., -(theta / 2.).sin()))
        }

        fn calculate_ccz(&self, qs: Array1<usize>) -> Array2<Complex<f64>> {
            let (gate_coeffs, destab_list, stab_list) = self.ccz_decomp(qs);
            self.calculate_matrix_from_decomp(gate_coeffs, destab_list, stab_list)
        }

        fn calculate_matrix_from_decomp(
            &self,
            gate_coeffs: Array1<Complex<f64>>,
            destab_list: Array2<bool>,
            stab_list: Array2<bool>,
        ) -> Array2<Complex<f64>> {
            // get matrices and add together...?
            gate_coeffs
                .iter()
                .zip(destab_list.rows().into_iter().zip(stab_list.rows()))
                .fold(
                    Array2::<Complex<f64>>::zeros((
                        2usize.pow(self.num_qubits as u32),
                        2usize.pow(self.num_qubits as u32),
                    )),
                    |mut acc, (p, (destabs, stabs))| {
                        let d = destabs.iter().enumerate().fold(
                            Array2::<Complex<f64>>::eye(2usize.pow(self.num_qubits as u32)),
                            |mut acc, (i, d)| {
                                if *d {
                                    println!(
                                        "row {:?}\n{:?}",
                                        self.stabilizer_simulator
                                            .storage
                                            .slice(self.stabilizer_simulator.destab_row(i)),
                                        row_to_matrix(
                                            self.stabilizer_simulator
                                                .storage
                                                .slice(self.stabilizer_simulator.destab_row(i))
                                        )
                                    );
                                    acc = acc.dot(&row_to_matrix(
                                        self.stabilizer_simulator
                                            .storage
                                            .slice(self.stabilizer_simulator.destab_row(i)),
                                    ));
                                }
                                acc
                            },
                        );
                        let s = stabs.iter().enumerate().fold(
                            Array2::<Complex<f64>>::eye(2usize.pow(self.num_qubits as u32)),
                            |mut acc, (i, s)| {
                                if *s {
                                    println!(
                                        "row {:?}\n{:?}",
                                        self.stabilizer_simulator
                                            .storage
                                            .slice(self.stabilizer_simulator.stab_row(i)),
                                        row_to_matrix(
                                            self.stabilizer_simulator
                                                .storage
                                                .slice(self.stabilizer_simulator.stab_row(i))
                                        )
                                    );
                                    acc = acc.dot(&row_to_matrix(
                                        self.stabilizer_simulator
                                            .storage
                                            .slice(self.stabilizer_simulator.stab_row(i)),
                                    ));
                                }
                                acc
                            },
                        );
                        println!("s: {:?}", s);
                        println!("d: {:?}", d);
                        println!("{:?}", d.dot(&s));
                        acc = acc + *p * d.dot(&s);
                        acc
                    },
                )
        }
    }

    #[test]
    fn t_gate_decomp_1q() {
        let NUM_QUBITS = 1;
        let MAX_BOND = 1;
        let mut stn = StabilizerTensorNetwork::new(NUM_QUBITS, MAX_BOND);
        // apply some random clifford gates
        stn.h(0);
        stn.s(0);
        stn.x(0);
        let calculated_t = stn.calculate_rz(PI / 4., 0);
        let actual_t = array![
            [1.0.into(), 0.0.into()],
            [0.0.into(), Complex::from_polar(1., PI / 4.)]
        ];

        println!("t: {actual_t}");
        println!("calculated t: {calculated_t}");
        assert_abs_diff_eq!(actual_t, calculated_t);
    }

    #[test]
    fn t_gate_decomp_2q() {
        let NUM_QUBITS = 2;
        let MAX_BOND = 2;
        let mut stn = StabilizerTensorNetwork::new(NUM_QUBITS, MAX_BOND);
        // apply some random clifford gates
        stn.h(0);
        stn.cnot(0, 1);
        stn.s(0);
        stn.x(0);
        let calculated_t = stn.calculate_rz(PI / 4., 0);
        let actual_t = ndarray::linalg::kron(
            &array![
                [1.0.into(), 0.0.into()],
                [0.0.into(), Complex::from_polar(1., PI / 4.)]
            ],
            &Array2::eye(2),
        );

        println!("t: {actual_t}");
        println!("calculated t: {calculated_t}");
        assert_abs_diff_eq!(actual_t, calculated_t);
    }

    #[test]
    fn t_gate_decomp_2q_b() {
        let NUM_QUBITS = 2;
        let MAX_BOND = 2;
        let mut stn = StabilizerTensorNetwork::new(NUM_QUBITS, MAX_BOND);
        // apply some random clifford gates
        stn.h(0);
        stn.cnot(0, 1);
        stn.s(0);
        stn.x(0);
        let calculated_t = stn.calculate_rz(PI / 4., 1);
        let actual_t = ndarray::linalg::kron(
            &Array2::eye(2),
            &array![
                [1.0.into(), 0.0.into()],
                [0.0.into(), Complex::from_polar(1., PI / 4.)]
            ],
        );

        println!("t: {actual_t}");
        println!("calculated t: {calculated_t}");
        assert_abs_diff_eq!(actual_t, calculated_t);
    }

    #[test]
    fn rz_gate_decomp_1q() {
        let NUM_QUBITS = 1;
        let MAX_BOND = 1;
        let theta = 1.;
        let mut stn = StabilizerTensorNetwork::new(NUM_QUBITS, MAX_BOND);
        // apply some random clifford gates
        stn.h(0);
        // stn.cnot(0, 1);
        stn.s(0);
        stn.x(0);
        let calculated_rz = stn.calculate_rz(theta, 0);
        let actual_rz = array![
            [1.0.into(), 0.0.into()],
            [0.0.into(), Complex::from_polar(1., theta)]
        ];

        println!("rz: {actual_rz}");
        println!("calculated rz: {calculated_rz}");
        assert_abs_diff_eq!(actual_rz, calculated_rz);
    }

    #[test]
    fn ccz_gate_decomp() {
        let mut stn = StabilizerTensorNetwork::new(3, 16);
        let mut actual_ccz = Array2::eye(8);
        actual_ccz[[7, 7]] = c64(-1., 0.);
        assert_abs_diff_eq!(actual_ccz, stn.calculate_ccz(array![0, 1, 2]));
        // apply some random clifford gates
        stn.h(0);
        stn.cnot(0, 1);
        stn.x(1);
        stn.s(0);
        assert_abs_diff_eq!(actual_ccz, stn.calculate_ccz(array![0, 1, 2]));
    }

    #[test]
    fn test_x_expectation_value() {
        let psi = StabilizerTensorNetwork::new(1, 1);
        let coeff = Complex::new(1., 0.);
        let stab = array![false];
        let destab = array![true];
        let eval = psi.evaluate_tn(coeff, stab, destab);

        assert_eq!(eval, Complex::new(0., 0.));
    }

    #[test]
    fn test_bell_state() {
        let mut psi = StabilizerTensorNetwork::new(3, 2);
        psi.h(0);
        psi.cnot(0, 2);
        psi.t(2);
        psi.cnot(0, 2);
        psi.h(0);
        let sv = psi.tensor_network.get_sv();
        println!("{:?}", sv);
        let eval = psi.expectation(vec![1, 0, 0], vec![0, 0, 0]);

        assert_abs_diff_eq!(eval, 0.5_f64.powf(0.5), epsilon = 1e-6);
    }

    #[test]
    fn test_bell_state_expectation() {
        let mut psi = StabilizerTensorNetwork::new(3, 2);
        psi.h(0);
        psi.cnot(0, 2);
        psi.t(2);
        let eval = psi.expectation(vec![1, 0, 1], vec![0, 0, 0]);

        assert_abs_diff_eq!(eval, 1., epsilon = 1e-6);
    }

    #[test]
    fn test_basic_projection() {
        let mut psi = StabilizerTensorNetwork::new(2, 2);
        println!("{}", psi.stabilizer_simulator);
        psi.cnot(0, 1);
        psi.h(0);
        println!("{}", psi.stabilizer_simulator);
        let outcome_one = psi.project(0);
        println!("{}", psi.stabilizer_simulator);
        println!("sv {:?}", psi.tensor_network.get_sv());
        if outcome_one == 0 {
            let z_expec = 1.0;
            let calculated_expec = psi.expectation(vec![1, 0], vec![0, 0]);
            assert_abs_diff_eq!(z_expec, calculated_expec);
        } else {
            let z_expec = -1.0;
            let calculated_expec = psi.expectation(vec![1, 0], vec![0, 0]);
            assert_abs_diff_eq!(z_expec, calculated_expec);
        }
    }

    #[test]
    fn test_projection() {
        let mut num_zeros = 0.;
        let NUM_SHOTS = 1000;
        for _ in 0..NUM_SHOTS {
            let mut psi = StabilizerTensorNetwork::new(3, 6);
            psi.h(0);
            psi.cnot(0, 1);
            psi.cnot(1, 2);
            psi.t(0);
            psi.t(1);
            psi.t(2);
            let outcome_one = psi.project(0);
            if outcome_one == 0 {
                num_zeros += 1.;
            }
            let outcome_two = psi.project(1);
            let outcome_three = psi.project(2);
            assert_abs_diff_eq!(outcome_one, outcome_two);
            assert_abs_diff_eq!(outcome_two, outcome_three);
        }
        assert_abs_diff_eq!(num_zeros / (NUM_SHOTS as f64), 0.5, epsilon = 1e-1);
    }

    #[test]
    fn test_ghz_state() {
        let mut psi = StabilizerTensorNetwork::new(3, 4);
        psi.h(0);
        psi.h(1);
        psi.h(2);
        psi.cnot(0, 1);
        psi.cnot(1, 2);
        psi.t(0);
        psi.t(1);
        psi.t(2);
        psi.cnot(1, 2);
        psi.cnot(0, 1);
        psi.h(0);
        psi.tensor_network.right_canonize_site(1);
        println!("schmidt {:?}", psi.tensor_network.schmidt_values(0));
        let sv = psi.tensor_network.get_sv();
        println!("{:?}", sv);
        println!("Bond Dimension is {}", psi.tensor_network.bond_dims);
        let eval = psi.expectation(vec![1, 0, 0], vec![0, 0, 0]);

        assert_abs_diff_eq!(eval, 0.3535533906, epsilon = 1e-6);
    }

    #[test]
    fn magic_state_injection() {
        let mut stn = MagicInjectedStn::new(1, 4, 0);
        stn.h(0);
        stn.t(0);
        stn.t(0);
        stn.t(0);
        stn.tdg(0);
        stn.s(0);
        stn.h(0);
        stn.project_magic();
        assert_abs_diff_eq!(-1., stn.expectation(vec![1], vec![0]), epsilon = 1e-6);
    }

    use proptest::{collection::vec, prelude::*};

    #[allow(unused)]
    #[derive(Debug, Clone)]
    enum CliffordGate {
        H(usize),
        X(usize),
        Y(usize),
        Z(usize),
        S(usize),
        Cnot(usize, usize),
    }

    fn arb_clifford(num_qubits: usize) -> impl Strategy<Value = CliffordGate> {
        prop_oneof![
            (0..num_qubits).prop_map(CliffordGate::H),
            (0..num_qubits).prop_map(CliffordGate::X),
            (0..num_qubits).prop_map(CliffordGate::Y),
            (0..num_qubits).prop_map(CliffordGate::Z),
            (0..num_qubits).prop_map(CliffordGate::S),
            (0..num_qubits, 0..num_qubits)
                .prop_filter("CNOT must be on different qubits", |(a, b)| a != b)
                .prop_map(|(c, t)| CliffordGate::Cnot(c, t)),
        ]
    }

    prop_compose! {
        fn arb_num_qubits(min_qubits: usize, max_qubits: usize)(num_qubits in min_qubits..max_qubits) -> usize {
            num_qubits
        }
    }

    prop_compose! {
        fn arb_clifford_vec(num_qubits: usize)(gates in vec(arb_clifford(num_qubits), 1..100)) -> Vec<CliffordGate> {
            gates
        }
    }

    prop_compose! {
        fn arb_stn(num_qubits: usize)(gates in arb_clifford_vec(num_qubits)) -> StabilizerTensorNetwork {
            let mut stn = StabilizerTensorNetwork::new(num_qubits, 100);
            for g in gates {
                println!("{:?}", g);
                match g {
                    CliffordGate::H(q) => stn.h(q),
                    CliffordGate::X(q) => stn.x(q),
                    CliffordGate::Y(q) => stn.y(q),
                    CliffordGate::Z(q) => stn.z(q),
                    CliffordGate::S(q) => stn.s(q),
                    CliffordGate::Cnot(c, t) => stn.cnot(c, t),
                }
            }
            stn
        }
    }

    fn stn_and_q() -> impl Strategy<Value = (StabilizerTensorNetwork, usize)> {
        (2..10usize).prop_flat_map(|num_qubits| (arb_stn(num_qubits), 0..num_qubits))
    }

    proptest! {
        #[test]
        fn calculate_t((stn, q) in stn_and_q()) {
            let calculated_t = stn.calculate_rz(PI / 4., q);
            let mut actual_t = Array2::eye(1);
            for i in 0..stn.num_qubits {
                if i == q {
                    actual_t = ndarray::linalg::kron(&actual_t, &array![[c64(1., 0.), c64(0., 0.)], [c64(0., 0.), Complex::<f64>::from_polar(1., PI/4.)]]);
                }
                else {
                    actual_t = ndarray::linalg::kron(&actual_t, &Array2::eye(2));
                }
            }
                assert_abs_diff_eq!(actual_t, calculated_t);
        }
    }
}
