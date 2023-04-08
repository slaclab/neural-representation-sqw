## Capturing dynamical correlations using implicit neural representations

<img width="1368" alt="Screen Shot 2023-03-21 at 11 15 28 PM" src="https://user-images.githubusercontent.com/39596225/226817874-f7c4009e-f892-4563-afff-4a8265b3639a.png">

---

## Installation

1) Make a new local folder and clone the repository

```
git clone https://github.com/src47/neural-representation-sqw.git
```

2) Install requirements

```
pip install -r requirements.txt
```

3) Make sure this repo directory is on the PYTHONPATH:

```bash
$ source shell/add_pwd_to_pythonpath.sh
```

## Directory Structure 

**data_experimental** 

This directory contains experiment S(q,w) measurements for both paths reported in the manuscript (with and without background subtraction). It also contains the accompanying energy and momentum coordinates.

**data_simulation_2023** 

Due to the size of the simulation dataset, it is not possible to inclue it directly on GitHub. Please download the simulation data which is publicly available at https://doi.org/10.5281/zenodo.7804447.

**notebooks** 

1) test_experimental_data.ipynb: contains code neccesary to optimize the surrogate implict neural model to fit experimental inelastic scattering data.  

2) test_low_counts.ipynb: contains code neccesary to fit experimental data as a function of count rate.

**models/siren** 

This directory contains a trained SIREN model which acts as a differentiable surrogate for linear spin wave simulations. 

## Training Model

To train the SIREN model on simulated excitations from a square lattice, please run:
```bash
$ python3 src/model_training.py --data_path data_simulation_2023/neural_dataset.npz
```

**Please direct any questions or comments to chitturi@stanford.edu, zhurun@stanford.edu, apetsch@stanford.edu, joshuat@slac.stanford.edu. 



