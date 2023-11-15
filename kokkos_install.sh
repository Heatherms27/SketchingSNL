#!/bin/bash

# Remove files, set directories and go to install location
rm -r CMakeCache.txt CMakeFiles cmake_install.cmake
KOKKOS_PATH=${HOME}/SketchingSNL/kokkos
KOKKOSKERNELS_PATH=${HOME}/SketchingSNL/kokkos-kernels
KOKKOS_SOURCE=$KOKKOS_PATH
KERNELS_SOURCE=$KOKKOSKERNELS_PATH
rm -r LibInstalls
mkdir LibInstalls
cd LibInstalls
mkdir kokkos-install
mkdir kernels-install
KOKKOS_INSTALL=${PWD}/kokkos-install
KERNELS_INSTALL=${PWD}/kernels-install

# Install Kokkos
cmake \
   -DCMAKE_CXX_COMPILER=${KOKKOS_PATH}/bin/nvcc_wrapper \
   -DCMAKE_INSTALL_PREFIX=$KOKKOS_INSTALL \
   -DKokkos_ENABLE_OPENMP=ON \
   -DKokkos_ENABLE_CUDA=ON \
   -DKokkos_ENABLE_CUDA_LAMBDA=ON \
   -DKokkos_ARCH_NATIVE=ON \
   $KOKKOS_SOURCE
make install -j 20

# Remove CMakeFiles from Kokkos install
rm -r CMakeFiles
rm CMakeCache.txt

# Build Kokkos Kernels
cmake \
   -DCMAKE_INSTALL_PREFIX=$KERNELS_INSTALL \
   -DKokkos_ROOT=$KOKKOS_INSTALL \
   -DCMAKE_CXX_COMPILER=${KOKKOS_PATH}/bin/nvcc_wrapper \
   -DKokkosKernels_ENABLE_TPL_BLAS=ON \
   -DKokkosKernels_ENABLE_TPL_LAPACK=ON \
   -DKokkosKernels_ENABLE_TPL_MKL=OFF \
   -DKokkosKernels_ENABLE_TPL_CUBLAS=ON \
   -DKokkosKernels_ENABLE_TPL_CUSPARSE=ON \
   $KERNELS_SOURCE
make install -j 20

# Return to application directory
cd ..

# Create application executable
cmake -DCMAKE_CXX_COMPILER=${KOKKOS_PATH}/bin/nvcc_wrapper -DKokkosKernels_DIR=$KERNELS_INSTALL/lib64/cmake/KokkosKernels .
make KOKKOS_DEVICES=Cuda,OpenMP -j 20

