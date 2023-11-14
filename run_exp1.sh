#!/bin/bash
#sbatch --exclusive examples/run_axonn_perlmutter.sh 2 2 2 16
sbatch --exclusive examples/run_axonn_perlmutter.sh 1 1 8 16
#sbatch --exclusive examples/run_axonn_perlmutter.sh 1 8 1 16
#sbatch --exclusive examples/run_axonn_perlmutter.sh 8 1 1 16
#sbatch --exclusive examples/run_axonn_perlmutter.sh 1 2 4 16
#sbatch --exclusive examples/run_axonn_perlmutter.sh 1 4 2 16
#sbatch --exclusive examples/run_axonn_perlmutter.sh 2 1 4 16
#sbatch --exclusive examples/run_axonn_perlmutter.sh 2 4 1 16
#sbatch --exclusive examples/run_axonn_perlmutter.sh 4 1 2 16
#sbatch --exclusive examples/run_axonn_perlmutter.sh 4 2 1 16
