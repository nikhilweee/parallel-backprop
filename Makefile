CXX = g++
CXXFLAGS = -Wall -g -O0 -std=c++11
CU = nvcc
CUFLAGS = -g -G -O0 -std=c++11

CPU_OUT = cpu.out
CUDA_OUT = cuda.out
TEST_OUT = tests.out

all: cpu cuda

clean: 
	$(RM) *.out

cpu: mlp.cc matrix.cc
	$(CXX) $(CXXFLAGS) -o $(CPU_OUT) mlp.cc matrix.cc

cuda: mlp.cc matrix.cu
	$(CU) $(CUFLAGS) -o $(CUDA_OUT) mlp.cc matrix.cu

cpu_tests: tests.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_OUT) tests.cc matrix.cc

cuda_tests: tests.cc
	$(CU) $(CUFLAGS) -o $(TEST_OUT) tests.cc matrix.cu


run_all:
	$(MAKE) run_cpu
	$(MAKE) run_cuda

run_cpu:
	$(MAKE) clean
	$(MAKE) cpu
	$(PWD)/$(CPU_OUT)

run_cuda:
	$(MAKE) clean
	$(MAKE) cuda
	$(PWD)/$(CUDA_OUT)

run_cpu_tests:
	$(MAKE) clean
	$(MAKE) cpu_tests
	$(PWD)/${TEST_OUT}

run_cuda_tests:
	$(MAKE) clean
	$(MAKE) cuda_tests
	$(PWD)/${TEST_OUT}

.SILENT:
