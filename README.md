## Magnetic excitations on the spin-1 square lattice: a continuous parameterization using implict neural representations

<img width="1368" alt="Screen Shot 2023-03-21 at 11 15 28 PM" src="https://user-images.githubusercontent.com/39596225/226817874-f7c4009e-f892-4563-afff-4a8265b3639a.png">

---

First make sure this repo directory is on the PYTHONPATH, e.g. by running:
```bash
$ source shell/add_pwd_to_pythonpath.sh
```

## Installation

1) Make a new local folder and clone the repository

```
git clone https://github.com/src47/neural-representation-sqw.git
```

2) Install requirements

```
pip install -r requirements.txt
```

## Directory Structure 

**data** 

**notebooks** 

1) test_experimental_data.ipynb: contains code neccesary to optimize the surrogate implict neural model to fit experimental inelastic scattering data.  

2) test_low_counts.ipynb: contains code neccesary to fit experimental data as a function of count rate.

**models** 

## Training Model

To train the SIREN model on simulated excitations from a square lattice, please run:
```bash
$ python3 src/model_training.py --data_path data_simulation_2023/neural_dataset.npz
```

**Please direct any questions or comments to chitturi@stanford.edu, zhurun@stanford.edu, apetsch@stanford.edu, joshuat@slac.stanford.edu. 



