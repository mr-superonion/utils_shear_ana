#!/usr/bin/env bash

xsubMini mpirun -n 28 merge_2pcf_mock.py --minId 0 --maxId 1404 --field XMM --datname $1 --mpi && sleep 0.5
xsubMini mpirun -n 28 merge_2pcf_mock.py --minId 0 --maxId 1404 --field VVDS --datname $1 --mpi && sleep 0.5
xsubMini mpirun -n 28 merge_2pcf_mock.py --minId 0 --maxId 1404 --field HECTOMAP --datname $1 --mpi && sleep 0.5
xsubMini mpirun -n 28 merge_2pcf_mock.py --minId 0 --maxId 1404 --field GAMA09H --datname $1 --mpi && sleep 0.5
xsubMini mpirun -n 28 merge_2pcf_mock.py --minId 0 --maxId 1404 --field WIDE12H --datname $1 --mpi && sleep 0.5
xsubMini mpirun -n 28 merge_2pcf_mock.py --minId 0 --maxId 1404 --field GAMA15H --datname $1 --mpi
