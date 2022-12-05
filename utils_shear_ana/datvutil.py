# Copyright 20220320 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
# python lib

import os
import numpy as np
import logging

import astropy.io.fits as pyfits

# Some useful functions for cosmosis

nzsDF = 4
nzallDF = (nzsDF + 1) * nzsDF // 2

# Functions to convert between HSC index order and COSMOSIS index order for
# covariance matrix:

# HSC year1 order:
# xip[z1,z1,theta1].. xip[z1,z1,thetaN] ..
# xim[z1,z1,theta1].. xim[z1,z1,thetaN] ..
# xip[z1,z2,theta1].. xip[z1,z1,thetaN] ..
# xim[z1,z2,theta1].. xim[z1,z1,thetaN] ..
# ......
# xip[zN,zN,theta1].. xip[z1,z1,thetaN] ..
# xim[zN,zN,theta1].. xim[z1,z1,thetaN] ..

# COSMOSIS order:
# xip[z1,z1,theta1].. xip[z1,z1,thetaN] ..
# xip[z1,z2,theta1].. xip[z1,z1,thetaN] ..
# ......
# xip[zN,zN,theta1].. xip[z1,z1,thetaN] ..
# ......
# xim[z1,z1,theta1].. xim[z1,z1,thetaN] ..
# xim[z1,z2,theta1].. xim[z1,z1,thetaN] ..
# ......
# xim[zN,zN,theta1].. xim[z1,z1,thetaN] ..


def make_empty_sep_mask(nzs, ntheta):
    """Makes an empty mask array with the default angular separation binning
    setup

    Args:
        nzs (int):      number of redshift binnings
        ntheta (int):   number of theta bins
    Returns:
        msks (dict):    a dictionary of mask arrays
    """
    types = [("xip", "?"), ("xim", "?")]
    mm0 = np.empty(ntheta, dtype=types)
    msks = {}
    for i in range(nzs):
        for j in range(i, nzs):
            mm = mm0.copy()
            mm["xip"] = np.ones(ntheta, dtype=bool)
            mm["xim"] = np.ones(ntheta, dtype=bool)
            msks.update({"%d%d" % (i + 1, j + 1): mm})
    return msks


def convert_cov_hsc2cosmosis(cov, nxp, nxm, nzall=nzallDF):
    """Converts a HSC covariance to COSMOSIS covariance (convariance estimated
    form mocks is stored in an different order)

    Args:
        cov (ndarray):  HSC covariance
        nxp,nxm (int):  number of xip, xim
        nzall (int):    number of cross correlations
    Returns:
        cov2 (ndarray): COSMOSIS covariance
    """
    assert cov.shape == ((nxp + nxm) * nzall, (nxp + nxm) * nzall)
    cov2 = np.zeros_like(cov)  # initialize the cosmosis covariance matrix
    for j in range(cov.shape[0]):
        for i in range(cov.shape[0]):
            j2 = convert_cov_index_cosmosis2hsc(j, nxp, nxm)
            i2 = convert_cov_index_cosmosis2hsc(i, nxp, nxm)
            cov2[j, i] = cov[j2, i2]
    return cov2


def convert_cov_index_cosmosis2hsc(ii, nxp, nxm, nzall=nzallDF):
    """Converts COSMOSIS covariance index to HSC covariance index [since
    convariance estimated form mocks is stored in an different order]

    Args:
        ii (int):       cosmosis index for column or row in cov matrix
        nxp,nxm (int):  number of xip, xim
        nzall (int):    number of cross correlations in redshift bins
    Returns:
        index (int)     HSC index for column or row in cov matrix
    """
    # 0 is xip
    # 1 is xim
    porm = int(ii >= (nxp * nzall))
    if porm == 0:
        nperiod = nxp
        ii_eff = ii
        pad = 0
    elif porm == 1:
        nperiod = nxm
        ii_eff = ii - (nxp * nzall)
        pad = nxp
    else:
        raise ValueError("input cosmosis index is not valid")
    jj = ii_eff // nperiod  # determines which redshift correlation bin (1..nzall)
    ji = ii_eff % nperiod  # determine the index of angular separation bin
    index = jj * (nxp + nxm) + pad + ji
    return index


def convert_treecor2cosmosis(corAll, mskAll, nzs=nzsDF):
    """Masks tpcfs and convert tpcfs into the cosmosis format

    Args:
        corAll (dict):  dictionary of tpcf
        mskAll (dict):  dictionary of mask for angular distance bin
    Returns:
        final (dict):   dictionary of cosmosis array
    """
    types = [
        ("BIN1", ">i8"),
        ("BIN2", ">i8"),
        ("ANGBIN", ">i8"),
        ("VALUE", ">f8"),
        ("ANG", ">f8"),
    ]
    final = {}
    for dname in ["xip", "xim"]:
        BIN1 = []
        BIN2 = []
        val = []
        angBin = []
        ang = []
        ndat = 0
        for i in range(nzs):
            for j in range(i, nzs):
                mskij = mskAll["%d%d" % (i + 1, j + 1)][dname]
                corij = corAll["%d%d" % (i + 1, j + 1)][dname][mskij]
                angij = corAll["%d%d" % (i + 1, j + 1)]["meanr"][mskij]
                nn = len(corij)
                BIN1.append(np.ones(nn) * (i + 1))
                BIN2.append(np.ones(nn) * (j + 1))
                val.append(corij)
                angBin.append(np.arange(1, nn + 1))
                ang.append(angij)
                ndat += nn
        out = np.empty(ndat, dtype=types)
        out["BIN1"] = np.hstack(BIN1)
        out["BIN2"] = np.hstack(BIN2)
        out["VALUE"] = np.hstack(val)
        out["ANGBIN"] = np.hstack(angBin)
        out["ANG"] = np.hstack(ang)
        final.update({dname: out})
        del out, BIN1, BIN2, val, angBin, ang, ndat
    return final


def make_nz_hdu(nzlst, zmid, zlow=None, zhigh=None):
    """Makes hdu for n(z) in cosmosis format [see
    https://github.com/joezuntz/2point/#readme]

    Args:
        nzlst (list):       list of nz_i(z) i=1..N bins
        zmid (ndarray):     1D redshift grids center
        zlow (ndarray):     1D redshift lower bounds (optional)
        zhigh (ndarray):    1D redshift higher bounds (optional)
    Returns:
        hud (hdu):          the hdu of number density as a function of redshift
    """
    hd = pyfits.Header()
    # Extent name
    hd["EXTNAME"] = "NZ_SAMPLE"
    # redshift Grids
    if (zlow is None) or (zhigh is None):
        dz = zmid[1] - zmid[0]
        assert np.all(
            np.abs(zmid[1:] - zmid[:-1] - dz) < dz / 10.0
        ), "grid is not uniform, please input zlow and zhigh"
        zlow = zmid - dz / 2.0
        zhigh = zmid + dz / 2.0
    c1 = pyfits.Column(name="Z_LOW", array=zlow, format="D")
    c2 = pyfits.Column(name="Z_MID", array=zmid, format="D")
    c3 = pyfits.Column(name="Z_HIGH", array=zhigh, format="D")
    cAll = [c1, c2, c3]
    n = len(nzlst)
    assert n > 0

    for i in range(n):
        cAll.append(pyfits.Column(name="BIN%d" % (i + 1), array=nzlst[i], format="D"))
    hdu = pyfits.BinTableHDU.from_columns(cAll, header=hd)
    return hdu


def make_cs_hdu(data, quant="G+R"):
    """Makes hdu for cosmic shear correlation functions in cosmosis format
    [see https://github.com/joezuntz/2point/#readme]
    The angular separations are in units of arcmin.

    Args:
        data (ndarray): two-point correlation function data
        quant (str):    'G+R' (for xi+) or 'G-R' (for xi-)
    Returns:
        t (hdu):        the hdu of two-point correlation for cosmic shear
    """
    assert tuple(data.dtype.names) == (
        "BIN1",
        "BIN2",
        "ANGBIN",
        "VALUE",
        "ANG",
    ), "Please make sure the input data has corrct colnames: \
                (BIN1, BIN2, ANGBIN, VALUE, ANG)"
    assert (quant == "G+R") | (
        quant == "G-R"
    ), "We only support xi+ or xi-, please set quant= G+R or G-R"
    hd = pyfits.Header()
    hd["2PTDATA "] = "T"
    hd["WINDOWS"] = "SAMPLE"
    hd["KERNEL_1"] = "NZ_SAMPLE"
    hd["KERNEL_2"] = "NZ_SAMPLE"
    if quant == "G+R":
        extname = "xi_plus"
    elif quant == "G-R":
        extname = "xi_minus"
    else:
        raise ValueError("%s is not supported. Quant should be G-R or G-R." % quant)
    hd["EXTNAME"] = extname
    hd["QUANT1"] = quant
    hd["QUANT2"] = quant

    t = pyfits.hdu.BinTableHDU(data=data, header=hd)
    t.header["TUNIT5"] = "arcmin"
    return t


def make_cscov_hdu(data, strtlst=[0], namelst=["xi_plus"]):
    """Makes hdu for covariance of cosmic shear correlation functions in
    cosmosis format
    [see https://github.com/joezuntz/2point/#readme]

    Args:
        data:       covariance matrix
        strtlst:    list of starting positions
        namelst:    list of names
    Returns:
        the hdu of covariance matrix for cosmic shear
    """
    assert len(strtlst) == len(
        namelst
    ), "The number of starting positions should equals the number \
                of names"
    nn = len(strtlst)

    hd = pyfits.Header()
    hd["EXTNAME"] = "COVMAT"

    for i in range(nn):
        hd["NAME_%d" % i] = namelst[i]
        hd["STRT_%d" % i] = strtlst[i]
    t = pyfits.hdu.ImageHDU(data=data, header=hd)
    return t


def get_cosmosis_cor(Dir, nn, jbin, ibin, do_mask=True):
    """Gets the theta and correlation function from cosmosis test outputs

    Args:
        Dir (str):        Parent directory for the cosmosis outputs
        nn (str):         corelation function name
        jbin,ibin (int):  tomographic redshift bin number
    Returns:
        theta (ndarray):  angular distance [arcmin]
        xi (ndarray):     correlation function
    """
    if not isinstance(nn, str):
        raise TypeError("nn should be string")
    if not isinstance(jbin, int) or not isinstance(ibin, int):
        raise TypeError("ibin and jbin should be integers")
    if nn not in ["plus", "minus"]:
        raise ValueError("nn should be either plus or minus")
    if jbin <= 0 or ibin <= 0:
        raise ValueError("ibin and jbin should be positive integers")
    # theta is in units of armin
    theta = (
        np.loadtxt(os.path.join(Dir, "shear_xi_%s/theta.txt" % nn))
        / np.pi
        * 180.0
        * 60.0
    )
    xi1 = np.loadtxt(os.path.join(Dir, "shear_xi_%s/bin_%d_%d.txt" % (nn, jbin, ibin)))
    if do_mask:
        msk = (theta > 1.5) & (theta < 360.0)
    else:
        msk = np.ones_like(theta).astype(bool)
    (theta, xi) = theta[msk], xi1[msk]
    return theta, xi


def get_tpcf_cosmosis(prefix, logrIn, Dir, nzs=nzsDF, dxi=None):
    """Gets the two-point correlaiton function from cosmosis outcome

    Args:
        prefix (str):           'plus' for xi_+ or 'minus' for xi_-
        logrIn (ndarray):       distance vector in log space
        Dir (str):              directory name of the cosmosis outcome
        nzs (int):              number of source redshift bins
        dxi (dict,ndarray):     systematic bias in Xi
    Returns:
        outP (hdu):             the hdu of twopoint correlation data vector
    """
    assert prefix in ["plus", "minus"], "prefix can only be plus or minus"
    assert os.path.isdir(Dir), "Cannot find directory: %s" % Dir

    infname = os.path.join(Dir, "shear_xi_%s/theta.txt" % prefix)
    thetaP = np.loadtxt(infname) / np.pi * 180.0 * 60.0
    dlogr = np.log(thetaP[1]) - np.log(thetaP[0])
    logrlow = np.log(thetaP) - dlogr / 2.0

    indP = np.int_((logrIn - logrlow[0]) / dlogr)
    # updates thetaP
    thetaP = thetaP[indP]
    # thetaP  =   np.exp(logrIn)
    nthetaP = len(thetaP)
    thetaBinP = np.arange(1, nthetaP + 1)
    ndataP = (nzs + 1) * nzs // 2 * nthetaP

    types = [
        ("BIN1", ">i8"),
        ("BIN2", ">i8"),
        ("ANGBIN", ">i8"),
        ("VALUE", ">f8"),
        ("ANG", ">f8"),
    ]
    outP = np.empty(ndataP, dtype=types)
    BIN1 = []
    BIN2 = []
    valP = []

    ic = 0
    for i in range(nzs):
        for j in range(i, nzs):
            BIN1.append(np.ones(nthetaP) * (i + 1))
            BIN2.append(np.ones(nthetaP) * (j + 1))
            vTmp = np.loadtxt(
                os.path.join(Dir, "shear_xi_%s/bin_%d_%d.txt" % (prefix, j + 1, i + 1))
            )[indP]
            if dxi is not None:
                if isinstance(dxi, dict):
                    vTmp = vTmp + dxi["%d%d" % (i + 1, j + 1)]
                else:
                    vTmp = vTmp + dxi[ic]
            valP.append(vTmp)
            ic += 1
            del vTmp
    ang = np.tile(thetaP, ic)
    angBin = np.tile(thetaBinP, ic)
    outP["BIN1"] = np.hstack(BIN1)
    outP["BIN2"] = np.hstack(BIN2)
    outP["ANGBIN"] = angBin
    outP["VALUE"] = np.hstack(valP)
    outP["ANG"] = ang
    return outP


def make_cosmosis_tpcf_hdulist_model(
    Dir, logrP, logrM=None, covmat=None, dxip=None, dxim=None
):
    """Makes hdu list for two point correlation functions and covariance from
    cosmosis output (model prediction) [see
    https://github.com/joezuntz/2point/#readme]

    Args:
        Dir (str):              directory of cosmosis prediction outcome
        logrP (ndarray):        the radius [in log] of sampling points for xi_p
        logrM (ndarray):        the radius [in log] of sampling points for xi_m
        covmat (ndarray):       covariance matrix
        dxip (ndarray, dict):   bias in xip
        dxim (ndarray, dict):   bias in xim
    Returns:
        hdul:       the hdulist of the data and covariance
    """
    if logrM is None:
        logrM = logrP.copy()
    nbinp = len(logrP)
    nbinm = len(logrM)
    if dxip is not None:
        if isinstance(dxim, dict):
            assert "11" in dxip.keys()
        elif isinstance(dxip, np.ndarray):
            assert dxip.shape[-1] == nbinp
        else:
            raise TypeError("dxip should be dict or ndarray")
    if dxim is not None:
        if isinstance(dxim, dict):
            assert "11" in dxim.keys()
        elif isinstance(dxim, np.ndarray):
            assert dxim.shape[-1] == nbinm
        else:
            raise TypeError("dxip should be dict or ndarray")

    tabP = get_tpcf_cosmosis("plus", logrP, Dir, dxi=dxip)
    tabM = get_tpcf_cosmosis("minus", logrM, Dir, dxi=dxim)

    hdu0 = pyfits.PrimaryHDU()
    if covmat is None:
        logging.warn("Do not have input covariance matrix, using a covariance matrix close to zero.")
        covmat = np.eye(len(tabP) + len(tabM)) * 1e-12
    assert covmat.shape == (len(tabP) + len(tabM), len(tabP) + len(tabM))
    hduCov = make_cscov_hdu(
        covmat, strtlst=[0, len(tabP)], namelst=["xi_plus", "xi_minus"]
    )
    hduP = make_cs_hdu(tabP, "G+R")
    hduM = make_cs_hdu(tabM, "G-R")
    hdul = pyfits.HDUList([hdu0, hduCov, hduP, hduM])
    return hdul


def make_cosmosis_tpcf_hdulist_data(corAll, mskAll, covmat, nzs=nzsDF):
    """Makes hdu list for two point correlation functions and covariance from
    data [see https://github.com/joezuntz/2point/#readme]

    Args:
        corAll (list):      list of correlations
        covmat (ndarray):    covariance matrix
    Returns:
        the hdulist of the data and covariance
    """
    dd = convert_treecor2cosmosis(corAll, mskAll, nzs)
    tabP = dd["xip"]
    tabM = dd["xim"]
    del dd

    hdu0 = pyfits.PrimaryHDU()
    if covmat is None:
        # make a diagonal, homogeneous covmat
        covmat = np.eye(len(tabP) + len(tabM)) * 1e-12
    assert covmat.shape == (len(tabP) + len(tabM), len(tabP) + len(tabM))
    hduCov = make_cscov_hdu(
        covmat, strtlst=[0, len(tabP)], namelst=["xi_plus", "xi_minus"]
    )
    hduP = make_cs_hdu(tabP, "G+R")
    hduM = make_cs_hdu(tabM, "G-R")
    hdul = pyfits.HDUList([hdu0, hduCov, hduP, hduM])
    return hdul


def get_cov_coeff(mat):
    """Gets the normalized correlation coefficent from the covariance matrix.

    Args:
        mat (ndarray):  input matrix
    Returns:
        out (ndarray):  normalized matrix
    """
    d = np.diag(mat) ** 0.5
    norm = np.tile(d, (len(d), 1))
    norm = norm * norm.T
    out = mat / norm
    return out


def make_conditional_datv(mu1, mu2, C11, C12, C21, C22, c1, seed=0):
    """Makes a random data vector 2 conditioned on data vector 1

    Args:
        mu1 (ndarray):  expectation of the first data vector
        mu2 (ndarray):  expectation of the first data vector
        C11 (ndarray):  first block covariance
        C12 (ndarray):  first-second block covariance
        C21 (ndarray):  second-first block covariance
        C22 (ndarray):  second-second block covariance
        c1 (ndarray):   data vector for 1
        seed (int):     random seed
    Returns:
        x2 (ndarray):   the realization second data vector
    """
    # check size
    n1 = C11.shape[0]
    n2 = C22.shape[0]
    if not C12.shape == (n1, n2):
        raise ValueError("Matrix C12 is in wrong shape.")
    if not C21.shape == (n2, n1):
        raise ValueError("Matrix C21 is in wrong shape.")
    if not c1.shape[-1] == n1:
        raise ValueError("the input data vector has a wrong shape.")
    # set seed
    np.random.seed(seed)
    c1 = np.atleast_2d(c1)
    size = c1.shape[0]  # number of realization

    # prepare
    iC11 = np.linalg.inv(C11)
    C22p = C22 - np.dot(C21, np.dot(iC11, C12))
    mu2p = np.zeros((size, n2))
    for i, _c1 in enumerate(c1):
        mu2p[i] = mu2 + np.dot(C21, np.dot(iC11, _c1 - mu1))
    # generate scatter
    n = np.random.multivariate_normal(np.zeros(n2), C22p, size=size)
    # add mean
    x2 = mu2p + n
    return x2
