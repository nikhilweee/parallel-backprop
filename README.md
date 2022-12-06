This repository implements a simple MLP in three languages: Python, C++ and CUDA.

# Python
It is recommended to use `conda` to install the required dependencies.

```console
$ conda env create -f env.yaml
```

To create the train and test datasets, and to start training, run the Python script.

```console
$ python network.py
```

# C++
To build the C++ project, simply run `make`.

```console
$ make run_cpu
```
