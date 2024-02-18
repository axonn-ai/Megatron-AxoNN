#!/bin/bash

userid=$(whoami)

if [ ! -d /mnt/bb/${userid}/axonn_venv ]; then
	cp /lustre/orion/scratch/${userid}/csc547/axonn_venv.tar.gz /mnt/bb/${userid}/
	cd /mnt/bb/${userid}/
	tar -xf axonn_venv.tar.gz
fi
