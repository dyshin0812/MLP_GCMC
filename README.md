This code was developed by the Jung-Hoon Lee group at Korea Institute of Science and Technology (KIST) for grand canonical Monte Carlo (GCMC) simulation based on machine learning potential (MLP).

This repository contains a modified implementation of the MLP-GCMC code originally developed by Goeminne et al. (J. Chem. Theory Comput. 2023, 19, 18, 6313–6325).
The original framework has been adapted to support interatomic potentials trained with SevenNet, enabling seamless integration of SevenNet-based models into GCMC simulations.
The MLP was trained using SevenNet (v0.11.2.post1), and the modified MLP-GCMC code has been tested with Python 3.12.4 and the Atomic Simulation Environment (ASE) v3.23.0.


Citation
If you use this modified code in your research, please cite the following papers:
1. Goeminne, R. et al. "DFT-Quality Adsorption Simulations in Metal–Organic Frameworks Enabled by Machine Learning Potentials," J. Chem. Theory Comput. 2023, 19 (18), 6313–6325.
2. "Unifying $CO_2$ Diffusion Mechanisms in Diamine-Functionalized Metal–Organic Frameworks via Quantum-Accurate Machine Learning Dynamics," under review.
