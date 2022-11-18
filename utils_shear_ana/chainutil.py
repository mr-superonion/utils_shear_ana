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

import scipy
import numpy as np
from scipy import stats
import scipy.optimize
import scipy.stats.kde as kde
from getdist import MCSamples
from chainconsumer import ChainConsumer
from tensiometer import gaussian_tension

import matplotlib.colors as mcolors
from cosmosis.output.text_output import TextColumnOutput

latexDict = {
    "omega_m": r"$\Omega_{\rm m}$",
    "omega_b": r"$\Omega_{\rm b}$",
    "sigma_8": r"$\sigma_8$",
    "s_8": r"$S_8$",
    "a_s": r"$A_s$",
    "n_s": r"$n_s$",
    "h0": r"$h_0$",
    "a": r"$A^{\mathrm{IA}}_1$",
    "a1": r"$A^{\mathrm{IA}}_1$",
    "a2": r"$A^{\mathrm{IA}}_2$",
    "alpha": r"$\eta^{\mathrm{IA}}_1$",
    "alpha1": r"$\eta^{\mathrm{IA}}_1$",
    "alpha2": r"$\eta^{\mathrm{IA}}_2$",
    "bias_ta": r"$b^{\mathrm{TA}}$",
    "alpha_psf": r"$\alpha^{\mathrm{psf}}$",
    "beta_psf": r"$\beta^{\mathrm{psf}}$",
    "psf_cor1_z1": r"$\alpha^{(2)}_1$",
    "psf_cor2_z1": r"$\beta^{(2)}_1$",
    "psf_cor3_z1": r"$\alpha^{(4)}_1$",
    "psf_cor4_z1": r"$\beta^{(4)}_1$",
    "psf_cor1_z2": r"$\alpha^{(2)}_2$",
    "psf_cor2_z2": r"$\beta^{(2)}_2$",
    "psf_cor3_z2": r"$\alpha^{(4)}_2$",
    "psf_cor4_z2": r"$\beta^{(4)}_2$",
    "psf_cor1_z3": r"$\alpha^{(2)}_3$",
    "psf_cor2_z3": r"$\beta^{(2)}_3$",
    "psf_cor3_z3": r"$\alpha^{(4)}_3$",
    "psf_cor4_z3": r"$\beta^{(4)}_3$",
    "psf_cor1_z4": r"$\alpha^{(2)}_4$",
    "psf_cor2_z4": r"$\beta^{(2)}_4$",
    "psf_cor3_z4": r"$\alpha^{(4)}_4$",
    "psf_cor4_z4": r"$\beta^{(4)}_4$",
    "bias_1": r"$\Delta{z}_1$",
    "bias_2": r"$\Delta{z}_2$",
    "bias_3": r"$\Delta{z}_3$",
    "bias_4": r"$\Delta{z}_4$",
    "m1": r"$\Delta{m}_1$",
    "m2": r"$\Delta{m}_2$",
    "m3": r"$\Delta{m}_3$",
    "m4": r"$\Delta{m}_4$",
    "logt_agn": r"$\Theta_{\mathrm{AGN}}$",
}
rangeDict = {
    "omega_m": [0.10, 0.45],
    "omega_b": [0.03, 0.06],
    "sigma_8": [0.60, 1.1],
    "s_8": [0.65, 0.9],
    "a_s": [0.1e-9, 6e-9],
    "n_s": [0.8, 1.15],
    "h0": [0.5, 0.9],
    "a": [-5, 5],
    "alpha": [-5, 5],
    "alpha_psf": [-0.04, 0.015],
    "beta_psf": [-4.0, 2.0],
    "logt_agn": [6.5, 8.5],
    "bias_1": [-0.1, 0.1],
    "bias_2": [-0.1, 0.1],
    "bias_3": [-0.1, 0.1],
    "bias_4": [-0.1, 0.1],
    "m1": [-0.1, 0.1],
    "m2": [-0.1, 0.1],
    "m3": [-0.1, 0.1],
    "m4": [-0.1, 0.1],
}

wmap13Dict = {
    "omega_m": 0.279,
    "omega_b": 0.046,
    "sigma_8": 0.82,
    "s_8": 0.79076,
    "a_s": 2.1841e-9,
    "n_s": 0.97,
    "h0": 0.7,
    "a": 0.95,
    "alpha": -1.84,
    "alpha_psf": 0.0,
    "beta_psf": 0.0,
    "logt_agn": 7.3,
}


def estimate_pvalue_from_chi2(chi2, dof):
    """Estimates pvalue from chi2 and degree of freedom

    Args:
        chi2 (float):   chi2
        dof (float):    effective degree of freedom
    Returns:
        p(float):       p-value
    """
    p = 1.0 - stats.chi2.cdf(chi2, round(dof))
    return p


def read_cosmosis_maxlike(infname):
    """Reads the maxlike output of cosmosis

    Args:
        infname (str):      file name
    Returns:
        out (ndarray):      chain
    """
    try:
        output_info = TextColumnOutput.load_from_options({"filename": infname})
    except:
        print("Cannot read file: %s" % infname)
        return None
    colnames, data, _, _, _ = output_info
    data = data[0]
    colnames = [c.lower().split("--")[-1] for c in colnames]
    nsample, npar = data.shape
    types = [tp for tp in zip(colnames, [">f8"] * npar)]
    cal_s8 = False
    if ("s_8" not in colnames) and ("omega_m" in colnames) and ("sigma_8" in colnames):
        types.append(("s_8", ">f8"))
        cal_s8 = True
    out = np.empty(nsample, dtype=types)
    for nm in colnames:
        out[nm] = data.T[colnames.index(nm)]

    assert out.dtype.names is not None
    if "weight" in out.dtype.names:
        out = out[out["weight"] > 0]
    if cal_s8:
        alpha = 0.5
        out["s_8"] = out["sigma_8"] * (out["omega_m"] / 0.3) ** alpha
    return out


def read_cosmosis_chain(infname, return_meta=False):
    """Reads the chain output of cosmosis

    Args:
        infname (str):          file name
        return_meta (bool):     whether return metadata
    Returns:
        out (ndarray): chain
    """
    try:
        output_info = TextColumnOutput.load_from_options({"filename": infname})
    except:
        print("Cannot read file: %s" % infname)
        return None
    colnames, data, metadata, _, final_meta = output_info
    metadata = metadata[0]
    final_meta = final_meta[0]
    metadata.update(final_meta)
    data = data[0]
    nsample, npar = data.shape
    # assert nsample==metadata['nsample']
    colnames = [c.lower().split("--")[-1] for c in colnames]
    types = [tp for tp in zip(colnames, [">f8"] * npar)]
    cal_s8 = False
    if ("s_8" not in colnames) and ("omega_m" in colnames) and ("sigma_8" in colnames):
        types.append(("s_8", ">f8"))
        cal_s8 = True
    out = np.empty(nsample, dtype=types)
    for nm in colnames:
        out[nm] = data.T[colnames.index(nm)]

    assert out.dtype.names is not None
    if "weight" in out.dtype.names:
        out = out[out["weight"] > 0]
    if cal_s8:
        alpha = 0.5
        out["s_8"] = out["sigma_8"] * (out["omega_m"] / 0.3) ** alpha

    for i in range(1, 6):
        inm = "bias_%d" % i
        if inm in colnames:
            out[inm] = -1 * out[inm]

    if return_meta:
        return out, metadata
    else:
        return out


def estimate_parameters_from_chain(infname, chain=None, ptype="max", do_write=True):
    """Estimate the parameters and write it to file

    Args:
        infname (str):      input file name for chain
        chain (ndarray):    input chain [default: None]
        do_write (bool):    whether write down the estimated parameters
    Returns:
        max_post (dict):    maximum a posterior
    """
    if chain is None:
        chain = read_cosmosis_chain(infname)
    if not isinstance(chain, np.ndarray):
        raise TypeError("chain should either be None or ndarray")

    outfname = infname[:-4] + "_%s.ini" % ptype
    names = list(chain.dtype.names)

    # remove those derived parameters
    for nn in ["prior", "like", "post", "weight"]:
        names.remove(nn)

    c = ChainConsumer()
    c.add_chain(
        [chain[nn] for nn in names],
        weights=chain["weight"],
        parameters=names,
        posterior=chain["post"],
        kde=1.2,
        name="",
        plot_point=False,
        color="black",
        shade_alpha=0.5,
        marker_alpha=0.5,
    )
    c.configure(
        global_point=True,
        statistics=ptype,
        label_font_size=20,
        tick_font_size=15,
        linewidths=0.8,
        legend_kwargs={"loc": "lower right", "fontsize": 20},
    )
    max_post = c.analysis.get_max_posteriors()

    if do_write:
        npar_write = 0
        lines = []
        start = False
        end = False
        with open(infname, "r") as infile:
            for _ in range(1000):
                ll = infile.readline()
                if ll == "## END_OF_VALUES_INI\n":
                    assert start, "not started yet?"
                    end = True
                    start = False
                    # print('end')
                if start:
                    tmp = ll.split("## ")[-1]
                    nn = tmp.split(" = ")[0]
                    if nn in names:
                        # write the estimated parameter to the string
                        tmp = "%s = %.6f\n" % (nn, max_post[nn])
                        npar_write += 1
                    lines.append(tmp)
                if end:
                    break
                if ll == "## START_OF_VALUES_INI\n":
                    start = True
                    # print('start')
        assert npar_write <= len(
            names
        ), "Not all the estimated parameters are writed into string, \
            something goes wrong?"
        with open(outfname, "wt") as outfile:
            outfile.write("".join(lines))
    return max_post


def get_neff_from_chain(data):
    """Gets the effective degrees of freedom from nested chain

    Args:
        data (ndarray):  nested chains
    Returns:
        neff (float):  effective degrees of freedom
    """
    names = list(data.dtype.names)
    for nn in names:
        if nn in ["like", "post", "weight", "prior", "s_8"]:
            names.remove(nn)
    c = MCSamples(
        samples=[data[nn] for nn in names],
        loglikes=data["like"],
        weights=data["weight"],
        names=names,
    )
    p = MCSamples(
        samples=[data[nn] for nn in names], names=names, weights=np.exp(data["prior"])
    )
    neff = gaussian_tension.get_Neff(c, prior_chain=p)
    return neff


class KDE(kde.gaussian_kde):
    def __init__(self, points, factor=1.0, weights=None):
        points = np.array(np.atleast_2d(points))
        self._factor = factor
        self.norms = []
        normalized_points = []

        for column in points:
            col_mean = column.mean()
            col_std = column.std()
            self.norms.append((col_mean, col_std))
            normalized_points.append((column - col_mean) / col_std)
        super(KDE, self).__init__(normalized_points, weights=weights)

    def covariance_factor(self):
        return self.scotts_factor() * self._factor

    def grid_evaluate(self, n, ranges):
        if isinstance(ranges, tuple):
            ranges = [ranges]
        slices = [slice(xmin, xmax, n * 1j) for (xmin, xmax) in ranges]
        grids = np.mgrid[slices]
        axes = [ax.squeeze() for ax in np.ogrid[slices]]
        flats = [
            (grid.flatten() - norm[0]) / norm[1]
            for (grid, norm) in zip(grids, self.norms)
        ]

        shape = grids[0].shape
        flats = np.array(flats)
        like_flat = self.evaluate(flats)
        like = like_flat.reshape(*shape)
        if len(axes) == 1:
            axes = axes[0]
        return axes, like

    def normalize_and_evaluate(self, points):
        points = np.array(
            [(p - norm[0]) / norm[1] for norm, p in zip(self.norms, points)]
        )
        return self.evaluate(points)


def std_weight(x, w):
    mu = mean_weight(x, w)
    r = x - mu
    return np.sqrt((w * r**2).sum() / w.sum())


def mean_weight(x, w):
    return (x * w).sum() / w.sum()


def median_weight(x, w):
    a = np.argsort(x)
    w = w[a]
    x = x[a]
    wc = np.cumsum(w)
    wc /= wc[-1]
    return np.interp(0.5, wc, x)


def percentile_weight(x, w, p):
    a = np.argsort(x)
    w = w[a]
    x = x[a]
    wc = np.cumsum(w)
    wc /= wc[-1]
    return np.interp(p / 100.0, wc, x)


def find_asymmetric_errorbars(levels, v, weights=None):
    N = len(v)

    # Generate and normalize weights
    if weights is None:
        weights = np.ones(N)
    weights = weights / weights.sum()

    # Normalize the parameter values
    mu = mean_weight(v, weights)
    sigma = std_weight(v, weights)
    x = (v - mu) / sigma

    # Build the P(x) estimator
    K = KDE(x, weights=weights)

    # Generate the axis over which get P(x)
    xmin = x[weights > 0].min()
    xmax = x[weights > 0].max()
    X = np.linspace(xmin, xmax, 500)
    Y = K.normalize_and_evaluate(np.atleast_2d(X))
    Y /= Y.max()

    peak1d = X[Y.argmax()]
    peak1d = peak1d * sigma + mu

    # Take the log but suppress the log(0) warning
    old_settings = np.geterr()
    np.seterr(all="ignore")
    Y = np.log(Y)
    np.seterr(**old_settings)  # reset to default

    # Calculate the levels
    def objective(level, target_weight):
        w = np.where(Y > level)[0]
        if len(w) == 0:
            weight_inside = 0.0
        else:
            low = X[w].min()
            high = X[w].max()
            inside = (x >= low) & (x <= high)
            weight_inside = weights[inside].sum()
        return weight_inside - target_weight

    limits = []
    for target_weight in levels:
        level = scipy.optimize.bisect(
            objective, Y[np.isfinite(Y)].min(), Y.max(), args=(target_weight,)
        )
        w = np.where(Y > level)[0]
        low = X[w].min()
        high = X[w].max()
        # Convert back to origainal space
        low = low * sigma + mu
        high = high * sigma + mu
        limits.append((low, high))
    return peak1d, limits


def smooth_likelihood(x, y, w=None, kde_factor=2.0):
    n = 100
    kde = KDE([x, y], factor=kde_factor, weights=w)
    mu_x = np.average(x, weights=w)
    mu_y = np.average(y, weights=w)
    if w is None:
        w = np.ones_like(x)
    dx = np.sqrt((w * (x - mu_x) ** 2.0).sum() / w.sum()) * 4.0
    dy = np.sqrt((w * (y - mu_y) ** 2.0).sum() / w.sum()) * 4.0
    # print(mu_x,mu_y,dx,dy)

    x_range = (max(x.min(), mu_x - dx), min(x.max(), mu_x + dx))
    y_range = (max(y.min(), mu_y - dy), min(y.max(), mu_y + dy))
    (x_axis, y_axis), like = kde.grid_evaluate(n, [x_range, y_range])
    return n, x_axis, y_axis, like


def find_contours(like, x, y, w, n, xmin, xmax, ymin, ymax, contour1, contour2):
    x_axis = np.linspace(xmin, xmax, n + 1)
    y_axis = np.linspace(ymin, ymax, n + 1)
    histogram, _, _ = np.histogram2d(
        x, y, bins=[list(x_axis), list(y_axis)], weights=w / np.nanmax(w)
    )

    def objective(limit, target):
        msk = np.where(like >= limit)
        count = histogram[msk]
        return count.sum() - target

    target1 = histogram.sum() * (1 - contour1)
    target2 = histogram.sum() * (1 - contour2)

    level1 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target1,))
    level2 = scipy.optimize.bisect(objective, like.min(), like.max(), args=(target2,))
    return level1, level2, like.sum()


def make_corner2D_plot(
    out,
    ax,
    name1="omega_m",
    name2="s_8",
    color="b",
    fill=False,
    label=None,
    kde_factor=2.0,
):
    ax.set_xlabel(latexDict[name1])
    ax.set_ylabel(latexDict[name2])
    # Get the data
    x = out[name1]
    y = out[name2]
    if "weight" in out.dtype.names:
        w = out["weight"]
    else:
        w = np.ones_like(out)
    assert (x.max() - x.min() > 0) & (y.max() - y.min() > 0)

    mu_x = np.average(x, weights=w)
    mu_y = np.average(y, weights=w)

    # using KDE
    n, x_axis, y_axis, like = smooth_likelihood(x, y, w, kde_factor)

    # Choose levels at which to plot contours
    contour1 = 1 - 0.68
    contour2 = 1 - 0.95
    level1, level2, _ = find_contours(
        like,
        x,
        y,
        w,
        n,
        x_axis[0],
        x_axis[-1],
        y_axis[0],
        y_axis[-1],
        contour1,
        contour2,
    )

    level0 = np.inf
    color2 = mcolors.ColorConverter().to_rgb(color)

    if fill:
        light = (color2[0], color2[1], color2[2], 0.6)
        dark = (color2[0], color2[1], color2[2], 0.2)
        ax.contourf(x_axis, y_axis, like.T, [level2, level0], colors=[light], alpha=0.3)
        ax.contourf(x_axis, y_axis, like.T, [level1, level0], colors=[dark], alpha=0.3)
    else:
        light = (color2[0], color2[1], color2[2], 0.8)
        dark = (color2[0], color2[1], color2[2], 0.5)
        ax.contour(x_axis, y_axis, like.T, [level2, level0], colors=[light])
        ax.contour(x_axis, y_axis, like.T, [level1, level0], colors=[dark])

    # Do the labels
    mu_x = np.average(x, weights=w)
    mu_y = np.average(y, weights=w)
    ax.plot(mu_x, mu_y, "x", c=color, markersize=10, label=label)
    return
