# GM-Koopman: Data-driven Modeling and Analysis of Granular Materials with Modern Koopman Theory

This repository contains the source code for an ongoing project on data-driven modeling of granular materials using the modern Koopman theory.

</br>

<p align="center">
  <img src="https://github.com/AtoosaParsa/GM-Koopman-public/blob/bba96f3debad48dd99438389b2492c4d053ff9d8/figures/Koopman_overview.png"  width="800">
</p>

</br>
</br>

## Installation
Consider using a dedicated virtual environment (conda or otherwise) with Python 3.9+ and install the [NeuroMANCER](https://github.com/pnnl/neuromancer) package:

```bash
pip install neuromancer
```


## Usage
The code can run on CPU but using a CUDA-enabled GPU is recommended. `DK_comp.py` is the main program for running an experiment with a two-particle system. The particle configuration is passed as a command line argument so execute `python DK_comp.py 0` to run an experiment for the first of the following four configurations:

<p align="center">
  <img src="https://github.com/AtoosaParsa/GM-Koopman-public/blob/bba96f3debad48dd99438389b2492c4d053ff9d8/figures/configurations.png"  width="600">
</p>

## Notes
This repository is part of an ongoing project and will be updated continuously.

Please do not hesitate to reach out directly or open a GitHub issue to start a conversation. Thank you for your interest in our work. 
