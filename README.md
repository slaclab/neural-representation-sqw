# Implicit Neural Representation for the dynamical structure factor ... 

<img width="500" alt="Screen Shot 2022-12-08 at 2 37 50 PM" src="https://user-images.githubusercontent.com/39596225/206581922-6fd39c22-79bf-417a-a94d-5eec1a705731.png">

This repository contains the trained models from the paper ... 

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



