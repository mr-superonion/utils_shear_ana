#!/usr/bin/env python

import os
import numpy as np
import astropy.io.fits as pyfits
from utils_shear_ana import datvutil, mea2pcf
from argparse import ArgumentParser

wrkDir = os.environ["rootDir"]

cminP = 2.94
cmaxP = 137.17
cminM = 12.89
cmaxM = 247.75

def make_model_mock(Dir, blind=0, num=0):
    """Gets logr bins for xip and xim

    Args:
        fname (str):    Cosmosis 2pt fits file name
        cdir (str):     Cosmosis output directory
        num (int):      number of simulations (if 0 make only noiseless one)
    """
    Dir = Dir.replace("./", "")
    assert "/" not in Dir, "do not support sub directory"

    blind_ver= "cat%d" %blind

    cor=mea2pcf.corDF
    rnom=cor.rnom

    mskp=(rnom>cminP)&(rnom<cmaxP)
    mskm=(rnom>cminM)&(rnom<cmaxM)

    # sampling points at logr
    logr1=cor.logr[mskp]
    logr2=cor.logr[mskm]

    # covariance
    cov= pyfits.getdata(
            os.path.join(
                wrkDir,"analysis/%s_xipm/data_extend_all_%s.fits"
                %(blind_ver,blind_ver)
                )
            )
    assert cov.shape==((len(logr1)+len(logr2))*10,(len(logr1)+len(logr2))*10)
    ofname=  '%s.fits' %Dir
    # msk array
    out   =  datvutil.make_cosmosis_tpcf_hdulist_model(Dir,logr1,logr2,cov)
    out.writeto(ofname,overwrite=True)
    nxp = len(out[2].data)
    nxm = len(out[3].data)

    if num > 0:
        odir = "%s_ran" %Dir
        os.makedirs(odir, exist_ok=True)
        datAll=np.hstack([out[2].data['value'],out[3].data['value']])
        mockAll=np.random.multivariate_normal(datAll, cov, num)

        # write output
        for i in range(num):
            ofname=  os.path.join(odir, "%s_ran%02d.fits" %(Dir,i))
            out2 = out.copy()
            out2[2].data['value']=mockAll[i,:nxp]
            out2[3].data['value']=mockAll[i,nxp:]
            out2.writeto(ofname,overwrite=True)
            del out2,ofname
    return


if __name__ == "__main__":
    parser = ArgumentParser(description="fpfs procsim")
    parser.add_argument(
        "dirname", type=str,
        help="cosmosis model file name",
        nargs='+',
    )
    parser.add_argument(
        "-b", "--blind",
        type=int, default=0,
        help="blinded version, 0, 1 or 2"
    )
    parser.add_argument(
        "-n", "--num",
        type=int, default=0,
        help="number of simlations"
    )
    args = parser.parse_args()
    for dd in args.dirname:
        make_model_mock(dd, args.blind, args.num)
