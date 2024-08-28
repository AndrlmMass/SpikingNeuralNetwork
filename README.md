# SpikingNeuralNetwork PhD project
This project aims to develop AGI-inspired SNN models that can perform online learning without labelling of data. It is based on a predictive coding inspired framework of the mind, and is the flagship project of the BONXAI group at NMBU.
### Disclaimer: 
Currently, the study is in its initial phase, developing a basic SNN model which will serve as the foundation of the model. 

#### Install repository

  $ git clone  https://github.com/AndrlmMass/SpikingNeuralNetwork.git

### Set up development environment

We have an `environment.yml` file that contains all packages to build and work with the script. Will develop a package later for testing.

Install the `conda` environment:

    $ conda env create --file environment.yml

Activate the `conda` environment:

    $ conda activate neuroai-dev

If the contents of the `environment.yml` are updated, update an existing environment with:

    $ conda env update --name neuroai-dev --file environment.yml --prune

Optionally, the dependencies can be installed independent of `conda` by using `pip` directly:

    $ python3 -m pip install -r requirements.txt
