#!/bin/bash

for i in 1 2 4 8 16
do
	for j in 1 2 4 8 16
	do
		for k in 1 2 4 8 16
		do
			product=$((i * j * k))

			if [ $product -eq 16 ]; then
				echo "Running for ($i, $j, $k)"
				sbatch --exclusive examples/run_axonn_perlmutter.sh $i $j $k 16
			fi
		done
	done
done
