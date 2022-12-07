This repository implements a simple MLP in three languages: Python, C++ and CUDA.

# Python

It is recommended to use `conda` to install the required dependencies.

```console
$ conda env create -f env.yaml
```

To create the train and test datasets, run `make_csv.py`

```console
$ python make_csv.py
```

To run backpropagation in Python, run `network.py`

```console
$ python network.py
```

# C++

To build the C++ version, use `make`.

```console
$ make run_cpu
```

You can also run tests using `make`.

```console
$ make run_cpu_tests
```

# CUDA

To build the CUDA version, use `make`.

```console
$ make run_cuda
```

You can also run tests using `make`.

```console
$ make run_cuda_tests
```