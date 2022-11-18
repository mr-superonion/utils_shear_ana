#!/usr/bin/env python

import numpy as np
import astropy.io.fits as pyfits
from utils_shear_ana import datvutil
from argparse import ArgumentParser

def make_model_mock(fname, cdir):
    """Gets logr bins for xip and xim

    Args:
        fname (str):    Cosmosis 2pt fits file name
        cdir (str):     Cosmosis output directory
    """
    # get the angular separations and the covariance
    hdulin=  pyfits.open(fname)

    # for pre1 the mask is the same for each cross zbins
    msk=(hdulin[2].data['bin1']==1)&(hdulin[2].data['bin2']==1)
    logr1=np.log(hdulin[2].data['ang'][msk])
    print("angular bins for Xip:")
    print("minimum and maximum scales [arcmin]:")
    print(np.exp(logr1.min()),np.exp(logr1.max()))
    print(len(logr1))

    msk=(hdulin[3].data['bin1']==1)&(hdulin[3].data['bin2']==1)
    logr2=np.log(hdulin[3].data['ang'][msk])
    print("angular bins for Xim:")
    print("minimum and maximum scales [arcmin]:")
    print(np.exp(logr2.min()),np.exp(logr2.max()))
    print(len(logr2))
    cov=hdulin[1].data
    out   =  datvutil.make_cosmosis_tpcf_hdulist_model(cdir,logr1,logr2,cov)
    return out


if __name__ == "__main__":
    parser = ArgumentParser(description="fpfs procsim")
    parser.add_argument(
        "--fitsname", required=True, type=str, help="cosmosis 2pt fits data"
    )
    parser.add_argument(
        "--cosoDir", required=True, type=str, help="cosmosis output directory"
    )
    parser.add_argument(
        "--outDir", required=True, type=str, help="output directory"
    )
