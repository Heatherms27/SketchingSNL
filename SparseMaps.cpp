/* Code to create a Sparse Maps sketching matrix */
/* Written by: Heather Switzer */
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sys/time.h>
#include <math.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <KokkosBlas1_sum.hpp>
#include <KokkosBlas1_nrm2_squared.hpp>
#include <KokkosBlas1_fill.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosBlas3_trsm.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosKernels_IOUtils.hpp>
#include <KokkosKernels_IOUtils.hpp>
#include <Kokkos_Random.hpp>
#include "SketchMatrix.h"
//using namespace std;

// Initialize data types for scalars, matrices, and vectors
using scalar_t = double;
using Layout = Kokkos::LayoutRight;   /* Column-major Order */
using ViewVectorType = Kokkos::View<scalar_t*, Layout>;
using ViewVectorTypeInt = Kokkos::View<int*, Layout>;
using ViewMatrixType = Kokkos::View<scalar_t**, Layout>;

void printHelp(){
  printf("HELP MESSAGE\n");
  printf("----------------------------------------------------------\n");
  printf("  -cols <Int>        Set the size of the matrix (Required)\n");
  printf("  -rows <Int>        Set the size of the embedding dimension (Required)\n");
  printf("  -nnzPerCol <Int>   Set the number of nonzeros per column \n");
  printf("  -seed <Int>        Set the random seed  \n");
  printf("  -help              Print this message \n");
}

int main(int argc, char* argv[])
{

  int seed      = -1;     /* Random Seed */
  int cols      = -1;     /* Size of the matrix sketching is beign applied to */
  int rows      = -1;     /* Sketch Size */
  int nnzPerCol = -1;     /* Number of nonzeros each column in the sparse matrix will hace */
  int i;                  /* Loop variable */

  /* Read in command line arguments */
  for (i = 0; i < argc; i++) 
  {
    if(!strcmp(argv[i], "-help")) {
      printHelp();
      return 0;
    } else if(!strcmp(argv[i], "-cols")) {
      cols = atoi(argv[i+1]);
      printf("Size of matrix: %d\n", cols);
    } else if (!strcmp(argv[i], "-rows")) {
      rows = atoi(argv[i+1]);
      printf("Size of sketch dimension: %d\n", rows);
    } else if (!strcmp(argv[i], "-nnzPerCol")) {
      nnzPerCol = atoi(argv[i+1]);
      printf("Number of nonzeros per column: %d\n", nnzPerCol);
    } else if (!strcmp(argv[i], "-seed")) {
      seed = atoi(argv[i+1]);
      printf("Random seed set to: %d\n");
    }
  }
  printf("-----------------------------------------------------\n");

  /* Check to make sure all arguments are valid */
  if (cols == -1) {
    printf("ERROR: Size of the matrix not given. Use '-cols' flag to declare a size.\n");
    return -1;
  } else if (rows == -1) {
    printf("ERROR: Size of the subspace embedding not given. Use '-rows' flag to declare a size.\n");
    return -1;
  } else if (nnzPerCol < 1 || nnzPerCol > rows) {
    printf("Nnz per column is either not set or invalid. Setting to default value (%d)\n", rows);
    nnzPerCol = rows;
  }
  if (seed < 1) {
    printf("The random seed is either not set or invalid. Setting to default value (1)\n");
    seed = 1;
  }
  printf("-----------------------------------------------------\n");

  // Insert random -1's and 1's into the sketching matrix
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution_row(0,rows-1);
  std::uniform_int_distribution<int> distribution_val(0,1);
  int randNum;

 /* Initialize Kokkos */
  Kokkos::initialize(argc, argv);
    
  // Set up the CSC for the SparseMaps matrix
  ViewVectorType    SVals("SVals", nnzPerCol*cols);
  ViewVectorTypeInt SRows("SRows", nnzPerCol*cols);
  
  Kokkos::parallel_for(nnzPerCol*cols, KOKKOS_LAMBDA(const int i) {
      randNum = distribution_val(generator);
      SVals(i) = 2*randNum-1;
  }); 

  Kokkos::fence();

  // Now insert indices of the rows for each column
  
  for (int i = 0; i < cols; i++) {
    Kokkos::parallel_for(nnzPerCol, KOKKOS_LAMBDA(const int j) {
      bool isUnique;

      do {
        randNum = distribution_row(generator);
        isUnique = true;
        for (int k = i*nnzPerCol; k < i*nnzPerCol + j; k++) {
          if (randNum == SRows(k)) {
            isUnique = false;
            break;
          }
        }
      } while (!isUnique);
      
      SRows(i*nnzPerCol+j) = randNum;  

    });
  }

  Kokkos::fence();

  Kokkos::finalize();

return 0;
}
