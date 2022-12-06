#!/usr/bin/env python

import os
import astropy.io.fits as pyfits
from utils_shear_ana import datvutil, mea2pcf
from argparse import ArgumentParser

wrkDir = os.environ["homeWrk"]

cminP = 2.94049251
cminM = cminP*4.38
cmaxP = 56.52
cmaxM = 247.75

def make_model_mock(Dir, blind=0):
    """Gets logr bins for xip and xim

    Args:
        fname (str):    Cosmosis 2pt fits file name
        cdir (str):     Cosmosis output directory
    """

    ver   =  "cut1"
    blind_ver= "cat%d" %blind

    cor=mea2pcf.corDF
    rnom=cor.rnom

    mskp=(rnom>cminP)&(rnom<cmaxP)
    mskm=(rnom>cminM)&(rnom<cmaxM)

    # sampling points at logr
    logr1=cor.logr[mskp]
    logr2=cor.logr[mskm]

    # covariance
    cov= pyfits.getdata(os.path.join(wrkDir,'cosmicShear/tpcf/%s/cov_%s.fits' %(blind_ver,ver)))
    assert cov.shape==((len(logr1)+len(logr2))*10,(len(logr1)+len(logr2))*10)
    ofname=  '%s.fits' %Dir
    # msk array
    out   =  datvutil.make_cosmosis_tpcf_hdulist_model(Dir,logr1,logr2,cov)
    out.writeto(ofname,overwrite=True)
    return


if __name__ == "__main__":
    parser = ArgumentParser(description="fpfs procsim")
    parser.add_argument(
        'dirname', type=str, help="cosmosis model file name", nargs='+',
    )
    parser.add_argument(
        "--blind", type=int, default=0, help="blinded version, 0, 1 or 2"
    )
    args = parser.parse_args()
    for dd in args.dirname:
        make_model_mock(dd, args.blind)
