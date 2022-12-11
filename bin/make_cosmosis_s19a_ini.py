#!/usr/bin/env python

import os
import yaml
from argparse import ArgumentParser
from utils_shear_ana import cosmosisutil


def main(datname, sampler):
    is_data =  datname in ["cat0", "cat1", "cat2"]
    if not is_data:
        assert os.path.isfile("sim/%s.fits" %datname), "cannot find data for simulations! "
    # os.system("cp $shear_utils/bin/shear_config ./")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("clusters/checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("stdout", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    setups_fname = os.path.join(os.environ["cosmosis_utils"], "config/s19a", "setups.yaml")
    with open(setups_fname) as file:
        setup_list = yaml.load(file, Loader=yaml.FullLoader)

    if is_data:
        for ss in setup_list[0:4]:
            for kk in ss.keys():
                cosmosisutil.make_config_ini(kk, datname, sampler, **ss[kk])
                break
    else:
        ss = setup_list[0]
        for kk in ss.keys():
            cosmosisutil.make_config_sim_ini(kk, datname, sampler, **ss[kk])
            break
    return


if __name__ == "__main__":
    parser = ArgumentParser(description="make_s19a_ini")
    parser.add_argument(
        "-d", "--datname", default="cat0", type=str,
        help="data name cat0, cat1, cat2 or cowls85"
    )
    parser.add_argument(
        "-s", "--sampler", default="multinest", type=str,
        help="sampler: multinest, minuit, multinest_final"
    )
    args = parser.parse_args()
    main(args.datname, args.sampler)
