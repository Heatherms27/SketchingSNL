cmake_minimum_required(VERSION 3.13)
project(KokkosSketching)

find_package(KokkosKernels REQUIRED)
find_package(Kokkos REQUIRED)

add_executable(sparsemaps SparseMaps.cpp)
target_link_libraries(sparsemaps Kokkos::kokkoskernels Kokkos::kokkos)

