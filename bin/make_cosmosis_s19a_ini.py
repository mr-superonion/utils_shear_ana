#!/usr/bin/env python

import os
from argparse import ArgumentParser
from utils_shear_ana import cosmosisutil


os.system("cp $shear_utils/bin/shear_config ./")
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("stdout", exist_ok=True)


cosmosisutil.make_config_ini("config_fid.ini")


if __name__ == "__main__":
    parser = ArgumentParser(description="make_s19a_ini")
    # parser.add_argument(
    #     "--outDir", required=True, type=str, help="output directory"
    # )
