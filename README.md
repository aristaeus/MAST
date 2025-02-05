## MAST - Magic State Injection Augmented Stabilizer Tensor Networks

This library forms the basis of the work in [arXiv:2411.12482](https://arxiv.org/abs/2411.12482) and includes the following simulators

1) A standard Matrix Product State (MPS) simulator
2) A stabilizer tableau simulator
3) A Stailizer Tensor Network (STN) simulator (based off [Phys. Rev. Lett (133) 230601](https://doi.org/10.1103/PhysRevLett.133.230601))
4) A Magic Injected Stabilizer Tensor Network (MAST) simulator (as introduced in [arXiv:2411.12482](https://arxiv.org/abs/2411.12482))

# Installation 

Before you get started, you will need the following instaled on your machine

1) [Rust](https://www.rust-lang.org/tools/install)
2) Python 3 and ```pip ```
3) ```maturin``` (you can install this by running ```pip install maturin```)
4) [OpenBLAS](http://www.openmathlib.org/OpenBLAS/docs/install)

You can then clone the repository and from that directory you can run ```pip install .``` to install the Python front-end. 

# Basic use

You can interact with all of the MPS, STN and MAST simulators in largely the same way. 

Here's a basic example

```
import mast
N = 50 # Number of qubits
t = 30 # Number of T gates
max_bond = 0 # Maximum bond dimension -- if 0 then no maximum is set
circ = mast.MagicInjectedStn(N, t, max_bond)
circ.h(0)
circ.cx(0,30)
circ.t(30)
# ... and so on ... #
# Before you can measure observables from MAST you must project the magic register
circ.project_magic()
# Now you can project the data register or get expectation values
circ.project(0)
```
