#!/bin/sh
#RUN BY ENTERING '. module_setup.sh'
module load sems-cmake/3.23.1
module load sems-gcc/10.1.0

#set OMP settings
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=64
