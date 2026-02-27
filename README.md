# MP2-libint2

A minimal implementation of second-order Møller-Plesset perturbation theory (MP2) based on the Libint2 integral library. This project is intended for learning and experimenting with quantum chemistry programming, including Hartree-Fock, AO integrals, AO-MO transformation, and MP2 energy evaluation.

This is the **developing branch**, which contains an actively evolving implementation with modularized code structure and experimental features.

---

# Features

Current capabilities include:

* Gaussian basis integral evaluation using Libint2
* Restricted Hartree-Fock (RHF) framework
* AO to MO integral transformation
* MP2 correlation energy calculation
* Modular structure suitable for learning and extension
* Simple Makefile-based build system
* Example molecular geometry files (.xyz), output files (.out) and Gaussian 16 results (.log)

---

# Project Structure

```
.
├── main.cpp          # Program entry point
├── integrals.cpp     # AO integral and MP2 energy evaluation using Libint2
├── ao2mo.cpp         # AO to MO integral transformation
├── hartree_fock.h    # Hartree-Fock related definitions
├── linalg.cpp        # Linear algebra utilities
├── globals.cpp       # Global variables and shared data
├── utils.cpp         # Utility functions
├── Makefile          # Build configuration
├── h2o.xyz           # Example molecule: water
├── DPPBz.xyz         # Example molecule: DPPBz
└── test/             # Test files and results
```

---

# Dependencies

You must install the following dependencies:

Required:

* C++ compiler supporting C++11 or later (g++, clang++)
* Libint2
* Eigen

---

# Installing Libint2

```

see https://github.com/evaleev/libint

```

---

# Build Instructions

modify the libint2 and Eigen path in Makefile and compile using:

```
make
```

This produces the executable:

```
MP2
```

---

# Running the Program

Example:

```
./MP2 h2o.xyz aug-cc-pVTZ
```

Or:

```
./MP2 DPPBz.xyz def2-SVP
```

The program will:

1. Read molecular geometry
2. Construct basis set
3. Compute AO integrals
4. Perform Hartree-Fock calculation
5. Transform integrals to MO basis
6. Compute MP2 correlation energy

---

# License

This project uses Libint2, which is distributed under GPL/LGPL licenses.

See:

```
LICENSE
COPYING
```

for details.

---

# Author

Xiaoyi Zhai

---

# Acknowledgements

* Libint2 by Edward F. Valeev
* Quantum chemistry open source community

---

# Notes

This code is intended primarily for educational and research learning purposes, not production-level calculations.
