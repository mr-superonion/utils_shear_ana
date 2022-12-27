#!/usr/bin/env python

import os
import yaml
import numpy as np
from argparse import ArgumentParser
from utils_shear_ana import cosmosisutil

setups_fname = os.path.join(os.environ["cosmosis_utils"], "config/s19a", "setups.yaml")
with open(setups_fname) as file:
    setup_list = yaml.load(file, Loader=yaml.FullLoader)
setup_names = [list(ss.keys())[0] for ss in setup_list]


def main(datname, sampler, sid, inds, num):
    # os.system("cp $shear_utils/bin/shear_config ./")
    # necessary directory to run cosmosis
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("clusters/checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("stdout", exist_ok=True)
    os.makedirs("configs", exist_ok=True)

    # data or simulation
    is_data =  datname in ["cat0", "cat1", "cat2"]
    if is_data:
        func = cosmosisutil.make_config_ini
        if num >= 0:
            raise ValueError("num should less than 0 to run on real data")
    else:
        func = cosmosisutil.make_config_sim_ini
        # number of simulations
        if num >= 0:
            datname = "%s_ran%02d" %(datname, num)
    if not is_data:
        assert os.path.isfile("sim/%s.fits" %datname), \
            "cannot find file %s.fits! please put simulation under ./sim/" %datname

    # get init file for every setup
    ll = [setup_list[i] for i in inds]
    for ss in ll:
        for kk in ss.keys():
            print("Writing config file for runname: %s" %kk)
            func(
                runname=kk, datname=datname,
                sampler=sampler, sid=sid, **ss[kk],
                )
            break
    return


if __name__ == "__main__":
    parser = ArgumentParser(description="make_s19a_ini")
    parser.add_argument(
        "-d", "--datname", default="cat0", type=str,
        help="data name cat0, cat1, cat2 or cowls85"
    )
    parser.add_argument(
        "-s", "--sampler", default="multinest", type=str, nargs='+',
        help="sampler: multinest, minuit, multinest_final"
    )
    parser.add_argument(
        "-r", "--runname", default="fid", type=str, nargs='+',
        help="runname index"
    )
    parser.add_argument(
        "-n", "--num",
        type=int, default=0,
        help="number of simlations"
    )
    args = parser.parse_args()

    # prepare runnames
    rnames = np.atleast_1d(args.runname)
    if not set(rnames) <= set(setup_names):
        print("%s is not in setup list" %set(rnames))
    inds = np.unique(np.array([ setup_names.index(rn) for rn in rnames ]))

    # prepare simulation list
    if args.num > 0:
        nlist = np.arange(args.num, dtype=int)
    else:
        # this is for real data
        nlist = np.array([-1])

    # prepare samplers
    samp_list = np.atleast_1d(args.sampler)
    for samp in samp_list:
        # iterate over samplers
        if samp[-1] in ['2', '3', '4']:
            sid =  eval(samp[-1])
        else:
            sid = 1
        for i in nlist:
            # iterate over id of simulations
            main(args.datname, samp, sid, inds, i)
