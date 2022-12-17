import numpy as np

# The following class are simplified version of Tianqing Zhang's PSF
# model code
def generate_delta_xip(params1, params2, sys_cors1, sys_cors2):
    """Generates the systematics error on xip using the parameters

    Args:
        params1 (ndarray):      alpha-like systematic parameters [first]
        params2 (ndarray):      alpha-like systematic parameters [second]
        sys_cors1 (ndarray):    systematic property matrix [first component]
        sys_cors2 (ndarray):    systematic property matrix [second component]
    Returns:
        delta_xip (ndarray):    the delta xip from the PSF systematic parameters
    """

    if not isinstance(sys_cors1, np.ndarray):
        raise TypeError("The input systematic matrix should be ndarray.")
    if not isinstance(sys_cors2, np.ndarray):
        raise TypeError("The input systematic matrix should be ndarray.")
    if not isinstance(params1, np.ndarray):
        raise TypeError("The input parameters should be ndarray.")
    if not isinstance(params2, np.ndarray):
        raise TypeError("The input parameters should be ndarray.")
    # sys_cors1 and sys_cors2 are in shape of (ncor, ncor, n_theta_bin)
    # for each theta bin it is
    # [
    # [ pp1 pq1 p1 ]
    # [ pq1 qq1 q1 ]
    # [ p1  q1  1  ]
    # ]
    # params is in shape of (nzs, ncor)
    # for each source redshift
    # [
    # alpha, beta, c1
    # ]
    # [
    # alpha, beta, c2
    # ]

    ncor = sys_cors1.shape[0]
    assert sys_cors1.shape[1] == ncor, "sys_cors1.shape is wrong."
    n_theta_bin = sys_cors1.shape[-1]
    assert sys_cors2.shape == (ncor, ncor, n_theta_bin), "sys_cors2.shape is wrong"
    if len(params1.shape) == 1:
        params1 = params1[None, :]
    if len(params2.shape) == 1:
        params2 = params2[None, :]

    nzs = params1.shape[0]
    if params1.shape != (nzs, ncor):
        raise ValueError(
            "The shape of parameters 1 (%s) are not correct" % (params1.shape)
        )
    if params2.shape != (nzs, ncor):
        raise ValueError(
            "The shape of parameters 2 (%s) are not correct" % (params2.shape)
        )

    return _generate_delta_xip(nzs, n_theta_bin,
                               params1, params2,
                               sys_cors1, sys_cors2,
                               )


def _generate_delta_xip(nzs, n_theta_bin, params1, params2, sys_cors1, sys_cors2):
    ncross = ((nzs + 1) * nzs) // 2  # number of cross-correlations
    delta_xip = np.zeros(shape=(ncross, n_theta_bin))
    zcs = 0  # id of the cross correlation (from 0 to ncross)
    for zi in range(nzs):
        for zj in range(zi, nzs):
            delta_xip[zcs] = np.dot(
                params1[zj], np.dot(params1[zi], sys_cors1)
            ) + np.dot(params2[zj], np.dot(params2[zi], sys_cors2))
            zcs += 1  # updates id
    return delta_xip


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


# utilities
def transform_sys_matrix(pp_corr, psf_const1, psf_const2):
    """Transforms systematic matrix from Tianqing's convention"""
    n_theta_bin = pp_corr[0][0].shape[0]
    ncor_tq = len(psf_const1)
    ncor = ncor_tq + 1
    sys_cor1 = np.zeros((ncor, ncor, n_theta_bin), dtype="<f8")
    sys_cor2 = np.zeros((ncor, ncor, n_theta_bin), dtype="<f8")
    for i in range(ncor_tq):
        for j in range(ncor_tq):
            sys_cor1[i, j] = pp_corr[i][j] / 2.0
            sys_cor2[i, j] = pp_corr[i][j] / 2.0
        sys_cor1[i, ncor_tq] = psf_const1[i]
        sys_cor1[ncor_tq, i] = psf_const1[i]

        sys_cor2[i, ncor_tq] = psf_const2[i]
        sys_cor2[ncor_tq, i] = psf_const2[i]

    for i in range(ncor_tq):
        for j in range(ncor_tq):
            assert np.all(sys_cor1[i, j] == sys_cor1[j, i])
            assert np.all(sys_cor2[i, j] == sys_cor2[j, i])
    return sys_cor1, sys_cor2


def transform_params(params_tq, nzs=4):
    """Transforms parameter from Tianqing's convention"""
    ncor_tq = 4
    ncshift = ncor_tq * nzs
    ncor = ncor_tq + 1
    params1 = np.zeros((nzs, ncor), dtype="<f8")
    params2 = np.zeros((nzs, ncor), dtype="<f8")

    for zi in range(nzs):
        for i in range(ncor_tq):
            params1[zi, i] = params_tq[i + ncor_tq * zi]
            params2[zi, i] = params_tq[i + ncor_tq * zi]
        params1[zi, ncor_tq] = params_tq[ncshift + 2 * zi]
        params2[zi, ncor_tq] = params_tq[ncshift + 2 * zi + 1]
    return params1, params2
