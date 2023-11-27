# This task measure the two point correlation function for real space
# cosmic shear
import gc
import os
import glob
import fitsio
import argparse
import treecorr
import numpy as np
from utils_shear_ana import mea2pcf
import astropy.table as astTable

ntheta = 17         # number of angular bins
corDF = treecorr.GGCorrelation(
    nbins=ntheta,
    min_sep=2.188,
    max_sep=332.954,
    sep_units="arcmin",
)

nzs =   4           # number of tomographic redshift bins
blind_ver='cat0'
cor =   mea2pcf.corDF
wrkDir= os.environ['homeWrk']

dataG=[]
calib_fname = os.path.join(
    os.environ['HOME'],
    "hsc_blinds/shear_biases_fiducial.csv",
)
mrTab=astTable.Table.read(calib_fname)

# collect data in 4 redshift bins
for i in range(4):
    fname = os.path.join(
        wrkDir,
        "S19ACatalogs/catalog_2pt/cat_fiducial_rmrg_zbin%s.fits" %(i+1)
    )
    dd = fitsio.read(fname)
    dataG.append(dd)


strTmp=os.path.join(wrkDir,'cosmicShear/tpcf/%s/cor%d%d_fiducial_rmrg.fits')
cors0={}
# Measure the correlation function
for i in range(nzs):
    cati= mea2pcf.convert_data2treecat(
        dataG[i],mrTab['m_shear_cat0'][i],
        mrTab['m_sel'][i],
        mrTab['a_sel'][i],
    )
    for j in range(i,nzs):
        ofname= strTmp%(blind_ver,i+1,j+1)
        if os.path.isfile(ofname):
            continue
        catj = mea2pcf.convert_data2treecat(
            dataG[j],
            mrTab['m_shear_cat0'][j],   # multiplicative bias
            mrTab['m_sel'][j],          # multplicative selection bias
            mrTab['a_sel'][j],          # additive selection bias
        )
        # correlation
        cor.clear()
        cor.process(cati,catj)
        cor.write(ofname)
        del catj
