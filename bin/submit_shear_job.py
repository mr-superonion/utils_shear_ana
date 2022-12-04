#!/usr/bin/env python

import os
import argparse

script="""
#!/bin/bash
#PBS -q %s
#PBS -l nodes=%d:ppn=%d
#PBS -m n
#PBS -V
#PBS -N %s
#PBS -o stdout/%s.o
#PBS -e stdout/%s.e
%s

export VAR=value

cd %s

echo "TASK starts at:"
date
%s
echo "TASK ends at:"
date
"""

# source ./shear_config

def submit_job(inifile, queue, jobname=None):
    if jobname is None:
        jobname = inifile.split(".ini")[0].split("config_")[-1]
    # assume to use maximum resource for each queue
    host = os.environ["HOSTNAME"][0:2]
    if host=="id":
        nodes_ppn = {
                "tiny":[1,1,1], "mini":[1,52,52],
                "mini_B":[1,52,52], "small":[4,52,208],
                "large":[20,52,1040]
                }[queue]
        walltime = ""
    elif host == "gw":
        nodes_ppn = {
                "tiny":[1,1,1], "mini":[1,28,28],
                "mini2":[4, 28, 112], "small":[1,504,504]
                }[queue]
        walltime = "#PBS -l walltime=7:00:00:00"
    elif host == "fe": # gfarm
        nodes_ppn = {
                "tiny":[1,1,1], "mini":[1,20,20],
                "small":[6,20,120]
                }[queue]
        walltime = ""
    else:
        raise ValueError("Does not support the currect server")

    if nodes_ppn[2] == 1:
        command = "cosmosis %s" %inifile
    else:
        command = "mpirun -n %d cosmosis --mpi %s"%(nodes_ppn[2], inifile)

    jobscript = script%(
            queue, nodes_ppn[0], nodes_ppn[1],
            jobname, jobname, jobname, walltime,
            os.getcwd(), command,
            )

    print(jobscript)

    with open("script_temp.sh", "w") as f:
        f.write(jobscript)

    os.system("qsub script_temp.sh")
    os.remove("script_temp.sh")
    return

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("inifile", type=str,
            help="path of config file")
    parser.add_argument("--queue", type=str,
            help="tiny, mini, small, large", default="mini"
            )
    parser.add_argument("--jobname", type=str,
            help="job name", default=None,
            )

    args = parser.parse_args()
    submit_job(args.inifile, args.queue, args.jobname)
