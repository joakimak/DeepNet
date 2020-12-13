# DeepNet

A neural network program in C++ which supports five different execution policies for the GEMM/dot operation used in both forward- and backpropagation:

* **Sequential** - Regular sequential execution. 
* **BlockTiled** - Sequential execution with block-tiled matrix multiplication.
* **StaticParallel** - CPU Parallel execution with a fixed number of threads.
* **DynamicParallel** - CPU Parallel execution with a dynamically sized thread pool.
* **Heterogeneous** - GPU Parallel execution. Appropriate grid- and block sizes are set automatically by the program.

Since ~97% of the total makespan is spent in GEMM/dot, parallelization of arithmetic operations has been left out. 

## Requirements

The program uses features from C++17. Other than that, any modern version of the following libraries need to be installed and setup properly:

* OpenMP
* CUDA

A build script for QMake is included in the repo.


## Performance

The following benchmarks were obtained using gcc with the -O3 optimization flag.

For a dense, 3-layer network with layer sizes of (128, 64, 10) trained for MNIST digit classification, I observed the following runtimes per mini-batch epoch of 256 samples:

* **Sequential** - 46 ms
* **BlockTiled** - 36 ms
* **StaticParallel** - 27 ms
* **DynamicParallel** - 8 ms
* **Heterogeneous** - 3 ms


