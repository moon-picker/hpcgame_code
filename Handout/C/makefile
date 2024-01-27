all: hello_cuda hello_mpi hello_omp

hello_cuda: hello_cuda.cu
	nvcc -o hello_cuda hello_cuda.cu

hello_mpi: hello_mpi.cpp
	mpic++ -o hello_mpi hello_mpi.cpp

hello_omp: hello_omp.cpp
	g++ -fopenmp -o hello_omp hello_omp.cpp -lgomp

clean:
	rm -f hello_cuda hello_mpi hello_omp