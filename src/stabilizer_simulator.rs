use ndarray::*;
use num::complex::{c64, Complex};

#[allow(unused)]
#[derive(Copy, Clone, Debug)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

#[allow(unused)]
impl Pauli {
    /// returns true for X or Y
    pub fn has_x(&self) -> bool {
        match self {
            Pauli::I | Pauli::Z => false,
            Pauli::X | Pauli::Y => true,
        }
    }

    /// returns true for Y or Z
    pub fn has_z(&self) -> bool {
        match self {
            Pauli::I | Pauli::X => false,
            Pauli::Y | Pauli::Z => true,
        }
    }
}

impl std::fmt::Display for StabilizerSimulator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stabs: ")?;
        for row in self.get_stabilizer_tableau().rows() {
            if row[2 * self.n] == 1 {
                write!(f, "-")
            } else {
                write!(f, "+")
            }?;
            for i in 0..self.n {
                if row[i] == 1 && row[i + self.n] == 0 {
                    write!(f, "Z")?;
                } else if row[i] == 0 && row[i + self.n] == 1 {
                    write!(f, "X")?;
                } else if row[i] == 1 && row[i + self.n] == 1 {
                    write!(f, "Y")?;
                } else {
                    write!(f, "I")?;
                }
            }
            write!(f, "\n       ")?;
        }
        write!(f, "\ndstbs: ")?;

        for row in self.get_destabilizer_tableau().rows() {
            if row[2 * self.n] == 1 {
                write!(f, "-")
            } else {
                write!(f, "+")
            }?;
            for i in 0..self.n {
                if row[i] == 1 && row[i + self.n] == 0 {
                    write!(f, "Z")?;
                } else if row[i] == 0 && row[i + self.n] == 1 {
                    write!(f, "X")?;
                } else if row[i] == 1 && row[i + self.n] == 1 {
                    write!(f, "Y")?;
                } else {
                    write!(f, "I")?;
                }
            }
            write!(f, "\n       ")?;
        }

        Ok(())
    }
}

/// Takes vectors in the form Z_1, ..., Z_n, X_1, ..., X_n and checks whether they commute
#[allow(unused)]
pub fn check_commutes(a: ArrayView1<i8>, b: ArrayView1<i8>) -> bool {
    let n = a.len() / 2;
    (Zip::from(a.slice(s![..n]))
        .and(b.slice(s![n..2 * n]))
        .fold(0, |acc, a, b| acc + a * b)
        + Zip::from(a.slice(s![n..2 * n]))
            .and(b.slice(s![..n]))
            .fold(0, |acc, a, b| acc + a * b))
        % 2
        == 0
}

/// Takes vectors in the form Z_1, ..., Z_n, X_1, ..., X_n, Phase and sets a = ab
pub fn multiply_pauli_inplace(a: &mut ArrayViewMut1<i8>, b: ArrayView1<i8>) {
    let n = a.len() / 2;
    // calculate overall phase accumulated
    let a_z = a.slice(s![..n]);
    let a_x = a.slice(s![n..2 * n]);
    let b_z = b.slice(s![..n]);
    let b_x = b.slice(s![n..2 * n]);
    let sign = ((2 * a[2 * n]
        + 2 * b[2 * n]
        + Zip::from(&a_z)
            .and(&a_x)
            .and(&b_z)
            .and(&b_x)
            .fold(0, |acc, &a_z, &a_x, &b_z, &b_x| {
                acc + if a_z == 0 && a_x == 0 {
                    0
                } else if a_z == 1 && a_x == 1 {
                    b_z - b_x
                } else if a_z == 0 && a_x == 1 {
                    b_z * (2 * b_x - 1)
                } else {
                    // a_z == 1 && a_x == 0
                    b_x * (1 - 2 * b_z)
                }
            }))
        % 4)
        / 2;

    // multiply paulis
    azip!((a in &mut *a, b in &b) *a ^= b);

    // update sign
    a[2 * n] = sign;
}

pub fn multiply_pauli_complex_phase(a: ArrayView1<i8>, b: ArrayView1<i8>) -> Complex<f64> {
    let phase_mat = array![
        [Complex::ONE, Complex::ONE, Complex::ONE, Complex::ONE],
        [Complex::ONE, Complex::ONE, -Complex::I, Complex::I],
        [Complex::ONE, Complex::I, Complex::ONE, -Complex::I],
        [Complex::ONE, -Complex::I, Complex::I, Complex::ONE]
    ];
    let n = a.len() / 2;
    let a_z = a.slice(s![..n]);
    let a_x = a.slice(s![n..2 * n]);
    let b_z = b.slice(s![..n]);
    let b_x = b.slice(s![n..2 * n]);
    c64(-1., 0.).powi(b[2 * n].into())
        * Zip::from(&a_z).and(&a_x).and(&b_z).and(&b_x).fold(
            Complex::ONE,
            |acc, &a_z, &a_x, &b_z, &b_x| {
                acc * phase_mat[[(2 * a_x + a_z) as usize, (2 * b_x + b_z) as usize]]
            },
        )
}

pub(crate) fn i_matrix() -> Array2<Complex<f64>> {
    array![[1.0.into(), 0.0.into()], [0.0.into(), 1.0.into()]]
}
pub(crate) fn z_matrix() -> Array2<Complex<f64>> {
    array![[1.0.into(), 0.0.into()], [0.0.into(), (-1.0).into()]]
}
pub(crate) fn x_matrix() -> Array2<Complex<f64>> {
    array![[0.0.into(), 1.0.into()], [1.0.into(), 0.0.into()]]
}

/// Takes a vector in the form Z_1, ..., Z_n, X_1, ..., X_n, Phase and converts it to a matrix in
/// the computational basis
#[allow(unused)]
pub(crate) fn row_to_matrix(row: ArrayView1<i8>) -> Array2<Complex<f64>> {
    let n = row.len() / 2;
    let zs = row.slice(s![0..n]);
    let xs = row.slice(s![n..2 * n]);
    let mut result = array![[1.0.into()]];
    for (z, x) in zs.iter().zip(xs.iter()) {
        let mut term = i_matrix();
        if *z == 1 {
            term = term.dot(&z_matrix());
        }
        if *x == 1 {
            term = term.dot(&x_matrix());
        }
        if *x == 1 && *z == 1 {
            term *= -Complex::I;
        }
        result = ndarray::linalg::kron(&result, &term);
    }
    c64(-1., 0.).powi(row[2 * n].into()) * result
}

#[derive(Clone, Debug)]
pub struct StabilizerSimulator {
    pub(crate) storage: Array2<i8>,
    pub n: usize,
}

#[allow(unused)]
impl StabilizerSimulator {
    pub fn new(n: usize) -> Self {
        StabilizerSimulator {
            storage: Array2::eye(2 * n + 1),
            n,
        }
    }

    pub fn add_qubit(&mut self) {
        let old_tableau = self.storage.clone();
        let n_prev = self.n;
        self.n += 1;
        self.storage = Array2::eye(2 * self.n + 1);

        // stabilizers
        self.storage
            .slice_mut(s![..n_prev, ..n_prev])
            .assign(&old_tableau.slice(s![..n_prev, ..n_prev]));
        self.storage
            .slice_mut(s![..n_prev, self.n..2 * self.n - 1])
            .assign(&old_tableau.slice(s![..n_prev, n_prev..2 * n_prev]));
        // destabilizers
        self.storage
            .slice_mut(s![self.n..2 * self.n - 1, ..n_prev])
            .assign(&old_tableau.slice(s![n_prev..2 * n_prev, ..n_prev]));
        self.storage
            .slice_mut(s![self.n..2 * self.n - 1, self.n..2 * self.n - 1])
            .assign(&old_tableau.slice(s![n_prev..2 * n_prev, n_prev..2 * n_prev]));
        // signs
        self.storage
            .slice_mut(s![..n_prev, 2 * self.n])
            .assign(&old_tableau.slice(s![..n_prev, 2 * n_prev]));
        self.storage
            .slice_mut(s![self.n..2 * n_prev, 2 * self.n])
            .assign(&old_tableau.slice(s![self.n..2 * n_prev, 2 * n_prev]));
    }

    /// Z stabilizer tableau
    fn zs_tableau(&self) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>> {
        s![..self.n, ..self.n]
    }

    /// X stabilizer tableau
    fn xs_tableau(&self) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>> {
        s![..self.n, self.n..2 * self.n]
    }

    /// Z destabilizer tableau
    fn zd_tableau(&self) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>> {
        s![self.n..2 * self.n, ..self.n]
    }

    /// X destabilizer tableau
    fn xd_tableau(&self) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>> {
        s![self.n..2 * self.n, self.n..2 * self.n]
    }

    /// Full Z tableau. Contains stabilizers _and_ destabilizers
    fn z_tableau(&self) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>> {
        s![..2 * self.n, ..self.n]
    }

    /// Full X tableau. Contains stabilizers _and_ destabilizers
    fn x_tableau(&self) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>> {
        s![..2 * self.n, self.n..2 * self.n]
    }

    /// Full column (stabilizers and destabilizers) at index q on Z
    fn z_col(&self, q: usize) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 1]>> {
        s![.., q]
    }

    /// Full column (stabilizers and destabilizers) at index q on X
    fn x_col(&self, q: usize) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 1]>> {
        s![.., self.n + q]
    }

    /// Full stabilizer row (z, x, sign) at index i
    pub(crate) fn stab_row(
        &self,
        i: usize,
    ) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 1]>> {
        s![i, ..]
    }

    /// Full destabilizer row (z, x, sign) at index i
    pub(crate) fn destab_row(
        &self,
        i: usize,
    ) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 1]>> {
        s![i + self.n, ..]
    }

    /// Sign Vector for Stabilizers
    fn signs_vec(&self) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 1]>> {
        s![..self.n, 2 * self.n]
    }

    /// Sign Vector for Destabilizers
    fn signd_vec(&self) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 1]>> {
        s![self.n.., 2 * self.n]
    }

    fn stabilizer_tableau(
        &self,
    ) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>> {
        s![..self.n, ..]
    }

    fn destabilizer_tableau(
        &self,
    ) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 2]>> {
        s![self.n..2 * self.n, ..]
    }

    /// Full Sign Vector
    fn sign_vec(&self) -> SliceInfo<[SliceInfoElem; 2], Dim<[usize; 2]>, Dim<[usize; 1]>> {
        s![.., 2 * self.n]
    }

    fn get_zs_tableau(&self) -> ArrayView2<i8> {
        self.storage.slice(self.zs_tableau())
    }

    fn get_xs_tableau(&self) -> ArrayView2<i8> {
        self.storage.slice(self.xs_tableau())
    }

    fn get_zd_tableau(&self) -> ArrayView2<i8> {
        self.storage.slice(self.zd_tableau())
    }

    fn get_xd_tableau(&self) -> ArrayView2<i8> {
        self.storage.slice(self.xd_tableau())
    }

    fn get_signs_vec(&self) -> ArrayView1<i8> {
        self.storage.slice(self.signs_vec())
    }

    fn get_stabilizer_tableau(&self) -> ArrayView2<i8> {
        self.storage.slice(self.stabilizer_tableau())
    }

    fn get_destabilizer_tableau(&self) -> ArrayView2<i8> {
        self.storage.slice(self.destabilizer_tableau())
    }

    // sets row a to ab
    /// WARNING: Possibly not safe to call in a stabilizer tensor network context: doesn't change
    /// tableau stably
    pub fn multiply_rows(&mut self, a: usize, b: usize) {
        // calculate overall phase accumulated
        let (mut a, b) = self.storage.multi_slice_mut((s![a, ..], s![b, ..]));
        multiply_pauli_inplace(&mut a, b.view());
    }

    /// WARNING: Possibly not safe to call in a stabilizer tensor network context: doesn't change
    /// tableau stably
    fn row_reduce(&mut self) {
        for i in 0..self.n {
            // get a z in the (j, j) position
            for j in i..self.n {
                if self.get_zs_tableau()[[j, j]] == 1 {
                    if i == j {
                        // can't borrow the same row twice below; and also no need to actually do
                        // the swap
                        break;
                    }
                    let (i, j) = self.storage.multi_slice_mut((s![i, ..], s![j, ..]));
                    Zip::from(i).and(j).for_each(std::mem::swap);
                    break;
                }
            }

            // row reduce
            for j in 0..self.n {
                if self.get_zs_tableau()[[j, i]] == 1 && i != j {
                    self.multiply_rows(j, i);
                }
            }
        }
    }

    pub fn x(&mut self, q: usize) {
        let (z, mut sign) = self
            .storage
            .multi_slice_mut((self.z_col(q), self.sign_vec()));
        azip!((r in &mut sign, z in &z) *r ^= z);
    }

    pub fn y(&mut self, q: usize) {
        self.x(q);
        self.z(q);
    }

    pub fn z(&mut self, q: usize) {
        let (x, mut sign) = self
            .storage
            .multi_slice_mut((self.x_col(q), self.sign_vec()));
        azip!((r in &mut sign, x in &x) *r ^= x);
    }

    pub fn h(&mut self, q: usize) {
        let (z, x, mut sign) =
            self.storage
                .multi_slice_mut((self.z_col(q), self.x_col(q), self.sign_vec()));

        azip!((r in &mut sign, x in &x, z in &z) *r ^= x * z);

        Zip::from(x).and(z).for_each(std::mem::swap);
    }

    pub fn s(&mut self, q: usize) {
        let (mut z, x, mut sign) =
            self.storage
                .multi_slice_mut((self.z_col(q), self.x_col(q), self.sign_vec()));
        azip!((r in &mut sign, z in &z, x in &x) *r ^= z * x);
        azip!((z in &mut z, x in &x) *z ^= x);
    }

    pub fn cnot(&mut self, control: usize, target: usize) {
        let (mut z_ctrl, x_ctrl, z_targ, mut x_targ, mut sign) = self.storage.multi_slice_mut((
            self.z_col(control),
            self.x_col(control),
            self.z_col(target),
            self.x_col(target),
            self.sign_vec(),
        ));
        azip!((r in &mut sign,
               z_ctrl in &z_ctrl,
               z_targ in &z_targ,
               x_ctrl in &x_ctrl,
               x_targ in &x_targ)
            *r ^= x_ctrl * z_targ * (x_targ ^ z_ctrl ^ 1));
        azip!((c in &mut z_ctrl, t in &z_targ) *c ^= t);
        azip!((c in &x_ctrl, t in &mut x_targ) *t ^= c);
    }

    pub fn measure(&mut self, q: usize) -> u8 {
        // check if there's a stabilizer we don't commute with
        if let Some(i) = self
            .get_xs_tableau()
            .column(q)
            .iter()
            .zip(0..)
            .filter(|(x, _)| **x != 0)
            .map(|(_, i)| i)
            .next()
        {
            // random outcome
            let result = rand::random::<u8>() % 2;

            // get only one row with x_q == 1
            for j in 0..2 * self.n {
                if j != i && self.storage[[j, q + self.n]] == 1 {
                    self.multiply_rows(j, i);
                }
            }

            // swap the destabilizer
            let (mut stab, mut destab) = self
                .storage
                .multi_slice_mut((self.stab_row(i), self.destab_row(i)));
            destab.assign(&stab);
            stab.fill(0);
            stab[2 * self.n] = result as i8;
            stab[q] = 1;
            // collapse
            result
        } else {
            // deterministic outcome
            self.storage.slice_mut(s![2 * self.n, ..]).fill(0);
            for j in 0..self.n {
                if self.get_xd_tableau()[[j, q]] == 1 {
                    self.multiply_rows(2 * self.n, j);
                }
            }
            self.storage[[2 * self.n, 2 * self.n]] as u8
        }
    }

    pub fn measure_all(&mut self) -> Vec<u8> {
        (0..self.n).map(|i| self.measure(i)).collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn init() {
        let sim = StabilizerSimulator::new(2);
        println!("z:\n{}", sim.get_zs_tableau());
        println!("x:\n{}", sim.get_xs_tableau());
        assert_eq!(sim.get_zs_tableau(), Array2::eye(2));
        assert_eq!(sim.get_xs_tableau(), Array2::zeros((2, 2)));
    }

    #[test]
    fn h_gate() {
        let mut sim = StabilizerSimulator::new(2);
        println!("z before:\n{}", sim.get_zs_tableau());
        println!("x before:\n{}", sim.get_xs_tableau());
        sim.h(0);
        println!("z after:\n{}", sim.get_zs_tableau());
        println!("x after:\n{}", sim.get_xs_tableau());
        assert_eq!(sim.get_zs_tableau(), array![[0, 0], [0, 1]]);
        assert_eq!(sim.get_xs_tableau(), array![[1, 0], [0, 0]]);
    }

    #[test]
    fn x_gate() {
        let mut sim = StabilizerSimulator::new(2);
        println!("z before:\n{}", sim.get_zs_tableau());
        println!("x before:\n{}", sim.get_xs_tableau());
        println!("sign before:\n{}", sim.get_signs_vec());
        sim.x(0);
        println!("z after:\n{}", sim.get_zs_tableau());
        println!("x after:\n{}", sim.get_xs_tableau());
        println!("sign after:\n{}", sim.get_signs_vec());
        assert_eq!(sim.get_zs_tableau(), Array2::eye(2));
        assert_eq!(sim.get_xs_tableau(), Array2::zeros((2, 2)));
        assert_eq!(sim.get_signs_vec(), array![1, 0]);
    }

    #[test]
    fn cnot_gate() {
        let mut sim = StabilizerSimulator::new(2);
        println!("z before:\n{}", sim.get_zs_tableau());
        println!("x before:\n{}", sim.get_xs_tableau());
        println!("sign before:\n{}", sim.get_signs_vec());
        sim.x(0);
        sim.cnot(0, 1);
        sim.row_reduce();
        println!("z after:\n{}", sim.get_zs_tableau());
        println!("x after:\n{}", sim.get_xs_tableau());
        println!("sign after:\n{}", sim.get_signs_vec());
        assert_eq!(sim.get_zs_tableau(), Array2::eye(2));
        assert_eq!(sim.get_xs_tableau(), Array2::zeros((2, 2)));
        assert_eq!(sim.get_signs_vec(), array![1, 1]);
    }

    #[test]
    fn measurement() {
        let mut sim = StabilizerSimulator::new(2);
        sim.x(0);
        assert_eq!(sim.measure(0), 1);
        assert_eq!(sim.measure(1), 0);
    }

    #[test]
    fn collapse_measurement() {
        let mut num_0 = 0;
        const NUM_SHOTS: u64 = 100;
        for _ in 0..NUM_SHOTS {
            let mut sim = StabilizerSimulator::new(2);
            sim.h(0);
            sim.cnot(0, 1);
            println!("z before:\n{}", sim.get_zs_tableau());
            println!("x before:\n{}", sim.get_xs_tableau());
            println!("z_destab before:\n{}", sim.get_zd_tableau());
            println!("x_destab before:\n{}", sim.get_xd_tableau());
            println!("sign before:\n{}", sim.get_signs_vec());
            println!("measurement: {}", sim.measure(0));
            println!("z after:\n{}", sim.get_zs_tableau());
            println!("x after:\n{}", sim.get_xs_tableau());
            println!("z_destab after:\n{}", sim.get_zd_tableau());
            println!("x_destab after:\n{}", sim.get_xd_tableau());
            println!("sign after:\n{}", sim.get_signs_vec());
            assert_eq!(sim.measure(0), sim.measure(1));
            if sim.measure(0) == 0 {
                num_0 += 1;
            }
        }
        assert_ne!(num_0, 0);
        assert_ne!(num_0, NUM_SHOTS);
    }

    #[test]
    fn test_check_commutes() {
        let sim = StabilizerSimulator::new(2);
        assert!(check_commutes(
            sim.storage.slice(s![0, ..-1]),
            array![1, 0, 0, 0].view()
        ));
        assert!(!check_commutes(
            sim.storage.slice(s![0, ..-1]),
            array![0, 0, 1, 0].view()
        ));
        assert!(check_commutes(
            sim.storage.slice(s![0, ..-1]),
            array![0, 1, 0, 1, 1].view()
        ));
    }

    #[test]
    fn test_row_to_matrix() {
        let x_row = array![0, 1, 0];
        let z_row = array![1, 0, 0];
        let xz_row = array![0, 1, 1, 0, 0];

        assert_eq!(row_to_matrix(x_row.view()), x_matrix());
        assert_eq!(row_to_matrix(z_row.view()), z_matrix());
        assert_eq!(
            row_to_matrix(xz_row.view()),
            ndarray::linalg::kron(&x_matrix(), &z_matrix())
        )
    }
}
