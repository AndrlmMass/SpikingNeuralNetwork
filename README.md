# SpikingNeuralNetwork PhD project
This project aims to replicate and expand the model developed in [Zenke et al.'s 2015 article](https://www.nature.com/articles/ncomms7922).

**Disclaimer**: Currently, the study is in its initial phase, developing a basic SNN model which will serve as the foundation of the model. 

## Install repository

    git clone  https://github.com/AndrlmMass/SpikingNeuralNetwork.git

## Set up development environment

### General

We have an `environment.yml` file that contains all packages to build and work with the script. Will develop a package later for testing.

**Install** the `conda` environment:

    conda env create --file environment.yml

**Activate** the `conda` environment:

    conda activate SNN_env_simpl

**Update** the `environment.yml` with existing env by:

    conda env update --name SNN_env_simpl --file environment.yml --prune'

### Noise & sleep

Navigate to the noise_analysis folder

```
cd noise_analysis
```

**Install** the `conda` environment:

    conda env create --file environment.yml

**Activate** the `conda` environment:

    conda activate noise_testing

**Update** the `environment.yml` with existing env by (if you add more libraries that are necessary for running the script):

    conda env update --name noise_testing --file environment.yml --prune'
