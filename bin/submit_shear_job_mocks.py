#!/usr/bin/env python

import os
import time
import argparse

script = """
#!/bin/bash
#PBS -q %s
#PBS -l nodes=%d:ppn=%d
#PBS -m n
#PBS -V
#PBS -N %s
#PBS -o stdout/%s.o
#PBS -e stdout/%s.e
%s
%s

export VAR=value

run_id=$(printf "%%02d" ${PBS_ARRAYID})
cd $PBS_O_WORKDIR

%s
"""

# source ./shear_config


def submit_job(inifile, min_id, max_id):
    jobname = "mock_%d_%d" % (min_id, max_id)
    # assume to use maximum resource for each queue
    host = os.environ["HOSTNAME"][0:2]
    if host == "id":  # idark
        queue = "mini"
        nodes_ppn = {
            "mini": [1, 52, 832],
        }[queue]
        walltime = ""
    elif host == "gw":  # gw
        queue = "mini"
        nodes_ppn = {
            "mini": [1, 28, 112],
        }[queue]
        walltime = "#PBS -l walltime=2:00:00:00"
    elif host == "fe":  # gfarm
        queue = "mini"
        nodes_ppn = {"mini": [1, 20, 120]}[queue]
        walltime = ""
    else:
        raise ValueError("Does not support the currect server")
    per_run = min(nodes_ppn[1], (max_id - min_id) + 1)
    queue_array = "#PBS -t %d-%d%%%d" % (min_id, max_id, per_run)
    command = "mpirun -n %d cosmosis %" % (
        nodes_ppn[1],
        inifile.replace("xx", "$run_id"),
    )
    jobscript = script % (
        queue,
        nodes_ppn[0],
        nodes_ppn[1],
        jobname,
        jobname,
        jobname,
        queue_array,
        walltime,
        command,
    )

    print(jobscript)
    time.sleep(0.5)

    with open("script_temp.sh", "w") as f:
        f.write(jobscript)
    os.system("qsub script_temp.sh")
    os.remove("script_temp.sh")
    return


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inifile",
        type=str,
        help="path of config file",
    )
    parser.add_argument(
        "--min_id",
        type=int,
        help="minimum id of the mocks",
    )
    parser.add_argument(
        "--max_id",
        type=int,
        help="maximum id of the mocks",
    )
    args = parser.parse_args()
    if not "xx" in args.inifile:
        raise ValueError("inifile does not have 'xx'")
    submit_job(args.inifile, args.min_id, args.max_id)
