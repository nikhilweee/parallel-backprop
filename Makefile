CXXFLAGS = -Wall -g -O0 -std=c++11
CPU = cpu.out
TEST = test.out

all: test cpu

run: run_cpu

clean: 
	$(RM) *.out

cpu: mlp.cc matrix.cc
	$(CXX) $(CXXFLAGS) -o $(CPU) mlp.cc matrix.cc

test: test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST) test.cc matrix.cc

run_all:
	$(MAKE) run_test
	$(MAKE) run_cpu

run_cpu:
	$(MAKE) clean
	$(MAKE) cpu
	$(PWD)/$(CPU)

run_test:
	$(MAKE) clean
	$(MAKE) test
	$(PWD)/${TEST}

.SILENT:
