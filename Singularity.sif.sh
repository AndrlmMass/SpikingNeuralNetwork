Bootstrap: docker
From: python:3.8

%post
    apt-get update && apt-get install -y some-dependency
    pip install numpy snntorch 
    conda install some-conda-package

%environment
    export PATH=/path/to/some/bin:$PATH
    export LD_LIBRARY_PATH=/path/to/some/lib:$LD_LIBRARY_PATH

%runscript
    exec python "$@"
