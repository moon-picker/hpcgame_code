cmake_minimum_required(VERSION 3.26)
project(moonpicker)
find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_STANDARD 12)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cuda_add_executable(hello_cuda hello_cuda.cu)
target_link_libraries(hello_cuda ${CUDA_LIBRARIES})
add_executable(hello_mpi hello_mpi.cpp)
target_link_libraries(hello_mpi MPI::MPI_CXX)
add_executable(hello_omp hello_omp.cpp)
target_link_libraries(hello_omp OpenMP::OpenMP_CXX)