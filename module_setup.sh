#!/bin/bash
#RUN BY ENTERING '. module_setup.sh'
module load cmake/3.25.1
module load gcc/12.2.0
module load cuda/12.0.0
module load openblas/0.3.23

#set OMP settings
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=64
