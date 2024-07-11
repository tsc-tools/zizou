# Zizou: time-series classification tools for Geoscience

Detecting unusual patterns in geo-monitoring data is a common task in geohazard monitoring. While the types of patterns can be very different between, for example, volcanic unrest and submarine landslides, the methodologies to detect these patterns are often very similar.
A typical workflow can be broken down as follows:

1.	Access waveform and sensor meta data.
2.	Divide continuous data into fixed-size windows.
3.	Compute features for each window such as statistical properties or spectrograms.
4.	Detect patterns in the feature time-series. This can be either done in a supervised (e.g. patterns preceding an eruption) or unsupervised fashion (e.g. cluster analysis).
5.	Visualise features and patterns on an interactive dashboard that updates as new data becomes available.

Zizou implements this workflow using modern, open-source Data Science tools. Several common features (e.g., spectrograms) and pattern recognition algorithms (e.g., deep autoencoders) are already part of the toolbox. More importantly, the toolbox was designed with extensibility in mind. Our long-term objective is to make zizou a platform for evaluating new algorithms for seismo-acoustic monitoring.

## Installation
To only compute features run:

```
pip install -U zizou
```

To also use the machine learning modules run:

```
pip install -U "zizou[ML]"
```
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