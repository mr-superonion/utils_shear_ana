#!/usr/bin/env python

import os
import yaml
from argparse import ArgumentParser
from utils_shear_ana import cosmosisutil


def main():
    # os.system("cp $shear_utils/bin/shear_config ./")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("stdout", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    setups_fname = os.path.join(os.environ["cosmosis_utils"], "config/s19a", "setups.yaml")
    with open(setups_fname) as file:
        setup_list = yaml.load(file, Loader=yaml.FullLoader)
    for ss in setup_list[0:1]:
        for kk in ss.keys():
            cosmosisutil.make_config_ini(kk, **ss[kk])
            break
    return




if __name__ == "__main__":
    parser = ArgumentParser(description="make_s19a_ini")
    # parser.add_argument(
    #     "--outDir", required=True, type=str, help="output directory"
    # )
    main()
