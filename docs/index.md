# Zizou: time-series classification tools for Geoscience

Zizou is a package to provide common tools for time-series classification in Geoscience. For example,
it allows you to download seismic data from an S3 bucket or an FDSN server, compute spectrograms, and then
use an Autoencoder to detect anomalies in the spectrograms.

## Requirements
* xarray
* boto3
* pandas
* numpy
* scipy
* obspy
* tqdm
* xarray
* pyyaml
* tonik

Using the machine learning modules requires the following additional packages:

* scikit-learn
* pytorch 

## Installation
To install zizou and its requirements (except for scikit-learn and pytorch) run:

```
pip install -U zizou
```

To install the machine learning requirements please refer to the respective installation
pages of [scikit-learn](https://scikit-learn.org/stable/install.html) and [pytorch](https://pytorch.org/get-started/locally/).

### Installation from source
#### Setup conda environment

```
cd zizou 
conda env create -f environment.yml
```

#### Install package in new environment
```
conda activate zizou 
cd zizou 
pip install -e .
```

## Run tests

To run only the quick tests:
```
cd zizou 
pytest
```
To run the whole test suite:
```
cd zizou 
pytest --runslow
```

## Setup Jupyter notebook kernel:

```
conda activate zizou 
python -m ipykernel install --user --name zizou 
kernda -o -y /path/to/jupyter/kernels/zizou/kernel.json
```

To find the path of your kernel.json file you can run:
```
jupyter --paths
```