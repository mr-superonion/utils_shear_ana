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
import treecorr
from . import catutil
from . import datvutil
import numpy as np
from scipy.interpolate import interp1d

"""nthetaDF (int): default number of angular bins"""
nthetaDF = 17
# corDF =   treecorr.GGCorrelation(nbins=nthetaDF,min_sep=0.25,max_sep=360.,sep_units='arcmin') # old one
"""corDF: defult correlation class"""
corDF = treecorr.GGCorrelation(
    nbins=nthetaDF, min_sep=2.188, max_sep=332.954, sep_units="arcmin"
)
"""rnomDF (ndarray): default angular bins"""
rnomDF = corDF.rnom
rminP = 7.13
rmaxP = 56.52
rminM = 31.28
rmaxM = 247.75
"""mskp (ndarray): mask for xip"""
mskp = (rnomDF > rminP) & (rnomDF < rmaxP)
"""mskp (ndarray): mask for xim"""
mskm = (rnomDF > rminM) & (rnomDF < rmaxM)

"""corP: correlation for PSF tests"""
corP = treecorr.GGCorrelation(nbins=30, min_sep=0.25, max_sep=360.0, sep_units="arcmin")
"""mskSys (ndarray): scale cut used for systematic tests"""
mskSys = (rnomDF > 2.0) & (rnomDF < 300.0)
"""rnomSys (ndarray): angular bins for systematic tests"""
rnomSys = rnomDF[mskSys]
"""corB360: correlation for B-mode tests"""
corB360 = treecorr.GGCorrelation(
    nbins=360, min_sep=0.21, max_sep=420.0, sep_units="arcmin"
)


class EBmode:
    def __init__(self, Dir, nzs=4, tmin=0.3, tmax=400.0):
        """Class to do EB mode separation based on equations (25) and (26) in
        https://arxiv.org/abs/astro-ph/0112441

        Args:
            Dir (str):      cosmosis output direcrory for theory extrapolation
                            e.g., @git_repo/config/mockana/halofit_nla_nopsf
            nzs (int):      number of redshift bins
            tmin (float):   minimum of angular scale, below which we use theory
            tmax (float):   maximum of angular scale, beyond which we use theory
        """
        self.Dir = Dir
        self.tmax = tmax
        self.tmin = tmin
        # preparation
        self.intp1 = {}  # xip
        self.intp2 = {}
        self.intm1 = {}  # xim
        self.intm2 = {}
        for ii in range(nzs):
            for jj in range(ii, nzs):
                bn = "%d%d" % (ii + 1, jj + 1)
                # xip
                theta, xim = datvutil.get_cosmosis_cor(
                    Dir, "minus", jj + 1, ii + 1, do_mask=False
                )
                # get dlnt
                lnt = np.log(theta)
                dlnt = lnt[1] - lnt[0]
                # mask the lower end of theta which is estimated from data
                msk = lnt >= np.log(self.tmax)
                theta = theta[msk]
                xim = xim[msk]
                self.intp1.update({bn: np.sum(xim * dlnt * 4.0)})
                self.intp2.update({bn: -np.sum(xim / theta**2.0 * dlnt * 12.0)})

                # xim
                theta, xip = datvutil.get_cosmosis_cor(
                    Dir, "plus", jj + 1, ii + 1, do_mask=False
                )
                # get dlnt
                lnt = np.log(theta)
                dlnt = lnt[1] - lnt[0]
                # mask the higher end of theta which is estimated from data
                msk = lnt <= np.log(self.tmin)
                theta = theta[msk]
                xip = xip[msk]
                self.intm1.update({bn: np.sum(xip * theta**2.0 * dlnt * 4.0)})
                self.intm2.update({bn: -np.sum(xip * theta**4.0 * dlnt * 12.0)})
        return

    def get_xipEB(self, corIn, bn, rsmth=0, rescale=True, structured=True):
        """Gets the Emode and Bmode for xip

        Args:
            corIn (ndarray):    correlation functions
            bn (str):           redshift bin name e.g., '11', '22', '12'
            rsmth (int):        smoothing scale
            rescale (bool):     whether rescaling the theory prediction to the data
            structured (bool):  whether return structured data
        Returns:
            out (ndarray):      array with the following three items
                                'rnom' -- angular scale bin
                                'xipE' -- E-mode for xip
                                'xipB' -- B-mode for xip
        """
        assert isinstance(rsmth, int), "rsmth should be int"
        rnom = corIn["r_nom"]
        assert rnom[-1] >= self.tmax, (
            "the upper-limit of input angular scale shall be greater than %.2f arcmin"
            % self.tmax
        )
        assert (
            rnom[0] < 300
        ), "the lower-limit of input angular scale shall be less than 300 arcmin"
        _m = rnom < self.tmax
        rnom = rnom[_m]
        lnt = np.log(rnom)
        dlnt = lnt[1] - lnt[0]
        xim = corIn["xim"][_m]
        xip = corIn["xip"][_m]
        if rescale:
            # rescaling factor
            msktmp = (rnom > 60) & (rnom < 300.0)
            tty, ximty = datvutil.get_cosmosis_cor(
                self.Dir, "minus", eval(bn[1]), eval(bn[0]), do_mask=False
            )
            mskty = (tty > 60.0) & (tty < 300.0)
            rs = np.average(xim[msktmp] * rnom[msktmp]) / np.average(
                ximty[mskty] * tty[mskty]
            )
            del msktmp, tty, ximty, mskty
        else:
            rs = 1.0
        # get EmB
        emb = 4.0 * dlnt * np.cumsum(xim[::-1])[::-1]
        _tmp = xim / rnom**2.0
        emb = emb - 12.0 * rnom**2.0 * dlnt * np.cumsum(_tmp[::-1])[::-1]
        emb = emb + ((self.intp1[bn] + rnom**2.0 * self.intp2[bn])) * rs
        xipE = (xip + xim + emb) / 2.0
        xipB = (xip - xim - emb) / 2.0

        # smoothing
        if rsmth > 1:
            rnom, xipE, xipB = self.smooth(rsmth, rnom, xipE, xipB)
        if structured:
            out = np.zeros(
                len(rnom),
                dtype=[
                    ("r_nom_%s" % bn, ">f8"),
                    ("xipe_%s" % bn, ">f8"),
                    ("xipb_%s" % bn, ">f8"),
                ],
            )
            out["r_nom_%s" % bn] = rnom
            out["xipe_%s" % bn] = xipE
            out["xipb_%s" % bn] = xipB
        else:
            out = np.stack([rnom, xipE, xipB])
        return out

    def get_ximEB(self, corIn, bn, rsmth=0, rescale=True, structured=True):
        """Gets the Emode and Bmode for xim

        Args:
            corIn (ndarray):    correlation functions [output of treecorr]
            bn (str):           redshift bin name e.g., '11', '22', '12'
            rsmth (int):        smoothing scale
            rescale (bool):     whether rescaling the theory prediction to the data
            structured (bool):  whether return structured data
        Returns:
            out (ndarray):      array with the following three columns
                                'rnom' -- angular scale bin
                                'xipE' -- E-mode for xip
                                'xipB' -- B-mode for xip
        """
        assert isinstance(rsmth, int), "rsmth should be int"
        rnom = corIn["r_nom"]
        assert rnom[0] <= self.tmin, (
            "the lower-limit of input angular scale shall be less than %.2f arcmin"
            % self.tmin
        )
        _m = rnom > self.tmin
        rnom = rnom[_m]
        lnt = np.log(rnom)
        dlnt = lnt[1] - lnt[0]
        xim = corIn["xim"][_m]
        xip = corIn["xip"][_m]

        if rescale:
            # rescaling factor
            msktmp = (rnom > 1.0) & (rnom < 20.0)
            tty, xipty = datvutil.get_cosmosis_cor(
                self.Dir, "plus", eval(bn[1]), eval(bn[0]), do_mask=False
            )
            mskty = (tty > 1.0) & (tty < 20.0)
            rs = np.average(xip[msktmp] * rnom[msktmp]) / np.average(
                xipty[mskty] * tty[mskty]
            )
            del msktmp, tty, xipty, mskty
        else:
            rs = 1.0

        emb = 4.0 * dlnt * np.cumsum(rnom**2.0 * xip) / rnom**2.0
        emb = emb - 12.0 * dlnt * np.cumsum(rnom**4.0 * xip) / rnom**4.0
        emb = emb + ((self.intm1[bn] / rnom**2.0 + self.intm2[bn] / rnom**4.0)) * rs
        # get ximE and ximB
        ximE = (xip + xim + emb) / 2.0
        ximB = (xip - xim + emb) / 2.0

        # smoothing
        if rsmth > 1:
            rnom, ximE, ximB = self.smooth(rsmth, rnom, ximE, ximB)
        if structured:
            out = np.zeros(
                len(rnom),
                dtype=[
                    ("r_nom_%s" % bn, ">f8"),
                    ("xime_%s" % bn, ">f8"),
                    ("ximb_%s" % bn, ">f8"),
                ],
            )
            out["r_nom_%s" % bn] = rnom
            out["xime_%s" % bn] = ximE
            out["ximb_%s" % bn] = ximB
        else:
            out = np.stack([rnom, ximE, ximB])
        return out

    def smooth(self, rsmth, rnom, xiE, xiB, backward=False):
        """Rebins the coorelation function (without repeatedly using a bin)

        Args:
            rsmth (int):    Smoothing length (in units of pixel)
            rnom (ndarray): Anuglar radius
            xiE (ndarray):  Emode
            xiB (ndarray):  Bmode
            backward (bool):stating backwardly?
        Returns:
            rnom (ndarray): smoothed anuglar radius
            xiE (ndarray):  smoothed Emode
            xiB (ndarray):  smoothed Bmode
        """
        assert rnom.shape == xiE.shape, "shape of rnom and xiE are not the same"
        assert rnom.shape == xiB.shape, "shape of rnom and xiB are not the same"
        if backward:
            rnom = np.flip(rnom)
            xiE = np.flip(xiE)
            xiB = np.flip(xiB)
        # First remove the bins (from large separations) if the number of bins
        # is not an integer times of `rsmth'
        nuse = rnom.shape[-1] // rsmth * rsmth
        xsh = rnom.shape[:-1] + (nuse // rsmth, rsmth)
        rtmp = np.delete(rnom, np.s_[nuse:], -1).reshape(xsh)
        # use rtmp as weight for the smoothing, which is not optimal
        rnom = np.exp(np.average(np.log(rtmp), axis=-1))
        # Then do the same thing for xiE and xiB
        etmp = np.delete(xiE, np.s_[nuse:], -1).reshape(xsh)
        xiE = np.average(etmp, weights=rtmp, axis=-1)
        btmp = np.delete(xiB, np.s_[nuse:], -1).reshape(xsh)
        xiB = np.average(btmp, weights=rtmp, axis=-1)
        if backward:
            rnom = np.flip(rnom)
            xiE = np.flip(xiE)
            xiB = np.flip(xiB)
        return rnom, xiE, xiB


def measure_2pcf_mock(datIn, mbias, msel=0.0):
    """Measures 2pcf from hsc mocks using treecorr

    Args:
        datIn (ndarray):    input mock catalog
        mbias (float):      average multiplicative bias (m+dm2)
        msel (float):       selection bias
        Returns:
        correlation function (treecorr object)
    """
    # a few galaxies with infinite |e| were found in the first version,
    # so let me keep this msk here for safety
    msk = (datIn["noise1_int"] ** 2.0 + datIn["noise2_int"] ** 2.0) < 10.0
    msk = msk & ((datIn["noise1_mea"] ** 2.0 + datIn["noise2_mea"] ** 2.0) < 10.0)
    datIn = datIn[msk]

    if not isinstance(mbias, (float, int)):
        raise TypeError("multiplicative shear estimation bias should be a float.")
    if not isinstance(msel, (float, int)):
        raise TypeError("multiplicative selection bias should be a float.")

    datIn = catutil.make_mock_catalog(datIn, mbias=mbias, msel=msel)

    tree_cat = convert_mock2treecat(datIn, mbias, msel)
    corDF.clear()
    corDF.process(tree_cat, tree_cat)
    return corDF.copy()


def convert_mock2treecat(datIn, mbias, msel=0.0, version="all"):
    """Converts HSC mock catalog to treecorr catalog

    Args:
        datIn (ndarray):    input mock catalog
        mbias (float):      average multiplicative bias (m+dm2)
        msel (float):       selection multiplicative bias [default=0]
        version (str):      the version of mock (all, shape or shear) [default="all"]
        Returns:
        treecorr catalog
    """
    g1I, g2I = catutil.get_shear_regauss_mock(datIn, mbias, msel, version)
    # g2 is sign-flipped for convention reason (+x is west for treecorr)
    tree_cat = treecorr.Catalog(
        g1=g1I,
        g2=-g2I,
        ra=datIn["ra_mock"],
        dec=datIn["dec_mock"],
        w=datIn["weight"],
        ra_units="deg",
        dec_units="deg",
    )
    return tree_cat


def measure_2pcf_data(datIn, mbias, msel=0.0, asel=0.0):
    """Measures 2pcf from HSC mocks (https://arxiv.org/pdf/1901.09488.pdf)
    using treecorr

    Args:
        datIn (ndarray):      input mock catalog
        mbias (float):      average multiplicative bias (m+dm2)
        msel (float):       selection multiplicative bias
        asel (float):       selection additive bias
        Returns:
        correlation function (treecorr object)
    """

    if not isinstance(mbias, (float, int)):
        raise TypeError("multiplicative shear estimation bias should be a float.")
    if not isinstance(msel, (float, int)):
        raise TypeError("multiplicative selection bias should be a float.")

    tree_cat = convert_data2treecat(datIn, mbias, msel, asel)
    corDF.clear()
    corDF.process(tree_cat, tree_cat)
    return corDF.copy()


def convert_data2treecat(datIn, mbias, msel=0.0, asel=0.0):
    """Converts HSC catalog to treecorr catalog

    Args:
        datIn (ndarray):    input mock catalog
        mbias (float):      average multiplicative bias (m+dm2)
        msel (float):       selection multiplicative bias
        asel (float):       selection additive bias
        Returns:
        tree_cat:           treecorr catalog
    """
    g1I, g2I = catutil.get_shear_regauss(datIn, mbias, msel, asel)
    tree_cat = treecorr.Catalog(
        g1=g1I,
        g2=-g2I,
        ra=datIn["i_ra"],
        dec=datIn["i_dec"],
        w=datIn["i_hsmshaperegauss_derived_weight"],
        ra_units="deg",
        dec_units="deg",
    )
    return tree_cat


# ---PSF ----
def convert_star2treecat(scat, types="P"):
    """Converts star catalog to treecorr catalog

    Args:
        scat (ndarray):  SC/LSST star catalog
        types (str):     'P', 'PQ' or 'PQR'
    Returns:
        treecat:    tuple of treecorr.Catalog catP, catQ, catR
    """
    treecat = []
    if "P" not in types:
        raise ValueError("types should be P , PQ or PQR")
    ra, dec = catutil.get_radec(scat)
    g1p, g2p = catutil.get_psf_ellip(scat)  # PSF shape
    g1p = g1p / 2.0
    g2p = g2p / 2.0  # from ellip to g
    catP = treecorr.Catalog(
        g1=g1p, g2=-g2p, ra=ra, dec=dec, ra_units="deg", dec_units="deg"
    )
    treecat.append(catP)
    if "PQ" in types:
        g1s, g2s = catutil.get_sdss_ellip(scat)  # star shape
        g1s = g1s / 2.0
        g2s = g2s / 2.0  # from ellip to g
        catQ = treecorr.Catalog(
            g1=g1p - g1s,
            g2=-(g2p - g2s),
            ra=ra,
            dec=dec,
            ra_units="deg",
            dec_units="deg",
        )
        treecat.append(catQ)
    if "PQR" in types:
        sstar = catutil.get_sdss_size(scat, "trace") ** 2.0
        spsf = catutil.get_psf_size(scat, "trace") ** 2.0
        fdt = (sstar - spsf) / sstar
        catR = treecorr.Catalog(
            g1=g1p * fdt, g2=-g2p * fdt, ra=ra, dec=dec, ra_units="deg", dec_units="deg"
        )
        treecat.append(catR)
    return tuple(treecat)


def measure_rho_simple(catP, catQ):
    """

    Args:
        catP (treecorr.Catalog):
                tree catalog for star shape
        catQ (treecorr.Catalog):
                tree catalog for star shape residual
    Returns:
        pp,pq,qq (treecorr.Correlation):
                three rho correlations of PSF
    """
    # pp
    corDF.clear()
    corDF.process(catP, catP)
    corpp = corDF.copy()
    # pq
    corDF.clear()
    corDF.process(catP, catQ)
    corpq = corDF.copy()
    # qq
    corDF.clear()
    corDF.process(catQ, catQ)
    corqq = corDF.copy()
    corDF.clear()
    return corpp, corpq, corqq


def measure_rho_all(catP, catQ, catR):
    """

    Args:
        catP (treecorr.Catalog):
                tree catalog for star shape
        catQ (treecorr.Catalog):
                tree catalog for star shape residual
        catR (treecorr.Catalog):
                tree catalog for star size residual
    Returns:
        pp,pq,pr,qq,qr,rr (treecorr.Correlation):
                six rho correlations of PSF
    """
    # pp
    corDF.clear()
    corDF.process(catP, catP)
    corpp = corDF.copy()
    # pq
    corDF.clear()
    corDF.process(catP, catQ)
    corpq = corDF.copy()
    # pr
    corDF.clear()
    corDF.process(catP, catR)
    corpr = corDF.copy()
    # qq
    corDF.clear()
    corDF.process(catQ, catQ)
    corqq = corDF.copy()
    # qr
    corDF.clear()
    corDF.process(catQ, catR)
    corqr = corDF.copy()
    # rr
    corDF.clear()
    corDF.process(catR, catR)
    corrr = corDF.copy()
    corDF.clear()
    return corpp, corpq, corpr, corqq, corqr, corrr


def estimate_alphabetaeta(gp, gq, gr, pp, pq, pr, qq, qr, rr):
    """Estimates alpha beta and eta

    Args:
        gp,gq,gr:   treecorr.Correlation
                    galaxy star shape correlations
        pp,pq,pr,qq,qr,rr:   treecorr.Correlation
                    star-star(psf) shape correlation
    Returns:
        alpha,beta (ndarray): alpha, beta parameters
        alphaA,betaA (float): average of alpha and beta parameters
    """
    ntheta = len(gp.xip)
    alpha = np.empty(ntheta)
    beta = np.empty(ntheta)
    eta = np.empty(ntheta)
    xlist = []
    ylist = []
    for i in range(ntheta):
        stdp = np.sqrt(gp.varxip[i])
        stdq = np.sqrt(gq.varxip[i])
        stdr = np.sqrt(gr.varxip[i])
        y = np.array([gp.xip[i] / stdp, gq.xip[i] / stdq, gr.xip[i] / stdr])
        x = np.array(
            [
                [pp.xip[i] / stdp, pq.xip[i] / stdp, pr.xip[i] / stdp],
                [pq.xip[i] / stdq, qq.xip[i] / stdq, qr.xip[i] / stdq],
                [pr.xip[i] / stdr, qr.xip[i] / stdr, rr.xip[i] / stdr],
            ]
        )
        out = np.linalg.lstsq(x, y, rcond=None)[0]
        alpha[i] = out[0]
        beta[i] = out[1]
        eta[i] = out[2]
        ylist.append(y)
        xlist.append(x)
        del x, y, stdp, stdq, out
    xA = np.vstack(xlist)
    yA = np.hstack(ylist)
    alphaA, betaA, etaA = np.linalg.lstsq(xA, yA, rcond=None)[0]
    return (alpha, beta, eta), (alphaA, betaA, etaA)


def estimate_alphabeta(gp, gq, pp, pq, qq):
    """Estimates alpha and beta

    Args:
        gp,gq:      treecorr.Correlation
            galaxy star shape correlations
        pp,pq,qq:   treecorr.Correlation
            star-star(psf) shape correlation
    Returns:
        alpha,beta: np.ndarray
            alpha, beta parameters
        alphaA,betaA: float
            average of alpha and beta parameters
    """
    ntheta = len(gp.xip)
    alpha = np.empty(ntheta)
    beta = np.empty(ntheta)
    for i in range(ntheta):
        y = np.array([gp.xip[i], gq.xip[i]])
        x = np.array([[pp.xip[i], pq.xip[i]], [pq.xip[i], qq.xip[i]]])
        xInv = np.linalg.inv(x)
        out = np.dot(xInv, y)
        alpha[i] = out[0]
        beta[i] = out[1]
        del x, y, xInv, out
    return alpha, beta


def estimate_alphabeta_list(gpl, gql, pp, pq, qq, msk):
    """Estimates alpha and beta

    Args:
        gpl,gql:    a list of treecorr.Correlation
                    galaxy star shape correlations
        pp,pq,qq:   treecorr.Correlation
                    star-star(psf) shape correlation
    Returns:
        alphaA,betaA (float):
                    average of alpha and beta parameters
    """
    xlist = []
    ylist = []
    for gp, gq in zip(gpl, gql):
        ntheta = len(gp.xip)
        assert len(msk) == ntheta
        for i in np.arange(ntheta)[msk]:
            stdp = np.sqrt(gp.varxip[i])
            stdq = np.sqrt(gq.varxip[i])
            y = np.array([gp.xip[i] / stdp, gq.xip[i] / stdq])
            x = np.array(
                [
                    [pp.xip[i] / stdp, pq.xip[i] / stdp],
                    [pq.xip[i] / stdq, qq.xip[i] / stdq],
                ]
            )
            ylist.append(y)
            xlist.append(x)
            del x, y, stdp, stdq
    xA = np.vstack(xlist)
    yA = np.hstack(ylist)
    alphaA, betaA = np.linalg.lstsq(xA, yA, rcond=None)[0]
    return alphaA, betaA


class pcaVector(object):
    def __init__(self, X=None, r=None):
        """This module builds the principal space for a list of vectors

        Args:
            X (ndarray):        input mean-subtracted data array, size=nobj times nx
        Atributes:
            bases (ndarray):    principal vectors
            ave (ndarray):      center, size=nx
            norm (ndarray):     normalization factor, size=nx
            stds (ndarray):     stds of the initializing data on theses axes
            projs (ndarray):    projection coefficients of the initializing data
        """

        if X is not None:
            # initialize with X
            assert len(X.shape) == 2
            nobj = X.shape[0]
            ndim = X.shape[1]
            if r is None:
                self.r = np.arange(ndim)
            else:
                assert len(r) == ndim
                self.r = r
            # subtract average
            self.ave = np.average(X, axis=0)
            X = X - self.ave
            # normalize data vector
            self.norm = np.sqrt(np.average(X**2.0, axis=0))
            X = X / self.norm
            self.data = X

            # Get covariance matrix
            Cov = np.dot(X, X.T) / (nobj - 1)
            # Solve the Eigen function of the covariance matrix
            # e is eigen value and eVec is eigen vector
            eVal, eVec = np.linalg.eigh(Cov)

            # The Eigen vector tells the combinations of these data vectors
            # Rank from maximum eigen value to minimum and only keep the first
            # nmodes
            bases = np.dot(eVec.T, X)[::-1]
            var = eVal[::-1]
            projs = eVec[:, ::-1]

            # remove those bases with extremely small stds
            msk = var > var[0] / 1e8
            self.stds = np.sqrt(var[msk])
            self.bases = bases[msk]
            self.projs = projs[:, msk]
            base_norm = np.sum(self.bases**2.0, axis=1)
            self.bases_inv = self.bases / base_norm[:, None]
        return

    def transform(self, X):
        """Transforms from data space to pc coefficients

        Args:
            X (ndarray): input data vectors [shape=(nobj,ndim)]
        Returns:
            proj (ndarray): projection array
        """
        assert len(X.shape) <= 2
        X = X - self.ave
        X = X / self.norm
        proj = X.dot(self.bases_inv.T)
        return proj

    def itransform(self, projs):
        """Transforms from pc space to data

        Args:
            projs (ndarray): projection coefficients
        Returns:
            X (ndarray): data vector
        """
        assert len(projs.shape) <= 2
        nm = projs.shape[-1]
        X = projs.dot(self.bases[0:nm])
        X = X * self.norm
        X = X + self.ave
        return X

    def write(self, fname):
        """Writes the principal component basis to disk. The saved attributes
        include bases, ave and norm.

        Args:
            fname (str):    the output file name
        """
        np.savez(fname, bases=self.bases, ave=self.ave, norm=self.norm, r=self.r)
        return

    def read(self, fname):
        """Initializes the class with npz file

        Args:
            fname (str):    the input npz file name
        """
        assert fname[-4:] == ".npz", (
            "only supports file ended with .npz, current is %s" % fname[-4:]
        )
        ff = np.load(fname)
        self.bases = ff["bases"]
        self.ave = ff["ave"]
        self.norm = ff["norm"]
        self.r = ff["r"]
        return
