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
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import matplotlib.lines as mlines
from matplotlib.colors import SymLogNorm
from chainconsumer import ChainConsumer
from nestcheck import data_processing, plots
from cosmosis.output.text_output import TextColumnOutput

from . import chainutil
from .datvutil import Interp1d

from matplotlib.ticker import Locator

# some default setups
kde0 = 1.
stat0 = "mean"
# list of blinded parameters
nlistb = ["s_8", "omega_m", "sigma_8", "a_s"]


chain_labels_dict = {
    "fid": "fiducial",
    "theta1": "include smaller scales",
    "theta2": "include larger scales",
    "theta3": r"$\theta_+<23'.3$, $\theta_-<102'$",
    "theta4": r"$\theta_+>17'.3$, $\theta_->75'.9$",
    "rmz1": r"remove $z$bin $1$",
    "rmz2": r"remove $z$bin $2$",
    "rmz3": r"remove $z$bin $3$",
    "rmz4": r"remove $z$bin $4$",
    "z1z2": r"$z$bins $1 \& 2$",
    "z3z4": r"$z$bins $3 \& 4$",
    "XMM": "XMM",
    "GAMA09H": "GAMA09H",
    "WIDE12H": "WIDE12H",
    "GAMA15H": "GAMA15H",
    "VVDS": "VVDS",
    "HECTOMAP": "HECTOMAP",
    "PSF": "no PSF",
    "dz": r"no $\Delta z$",
    "dm": r"no $\Delta m$",
    "demp": r"\texttt{DEmpZ}",
    "dnnz": r"\texttt{dNNz}",
    "mizuki": r"\texttt{mizuki}",
}

latexDict = {
    "omega_m": r"$\Omega_{\mathrm{m}}$",
    "omega_b": r"$\Omega_{\mathrm{b}}$",
    "ombh2": r"$\omega_{\mathrm{b}}/\!10^{-3}$",
    "sigma_8": r"$\sigma_8$",
    "s_8": r"$S_8$",
    "a_s": r"$A_s /\!10^{-9}$",
    "n_s": r"$n_s$",
    "h0": r"$h_0$",
    "w": r"$w$",
    "mnu": r"$m_\nu$",
    "a": r"$A_1$",
    "a1": r"$A_1$",
    "a2": r"$A_2$",
    "alpha": r"$\eta_1$",
    "alpha1": r"$\eta_1$",
    "alpha2": r"$\eta_2$",
    "bias_ta": r"$b_{\mathrm{ta}}$",
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
    "a_bary": r"$A_{\mathrm{b}}$",
    "psf_alpha2": r"$\alpha^{(2)}$",
    "psf_beta2": r"$\beta^{(2)}$",
    "psf_alpha4": r"$\alpha^{(4)}$",
    "psf_beta4": r"$\beta^{(4)}$",
}

rangeDict = {
    "omega_m": [0.10, 70],
    "ombh2": [20, 25],
    "mnu": [0.06, 0.6],
    "w": [-2, -1.0 / 3.0],
    "a_s": [0.5, 10],
    "n_s": [0.86, 1.07],
    "h0": [0.62, 0.82],
    "a_bary": [2.0, 3.13],
    "logt_agn": [7.2, 8.3],
    "a": [-6, 6],
    "a1": [-6.0, 6.0],
    "a2": [-6.0, 6.0],
    "alpha": [-6.0, 6.0],
    "alpha1": [-6.0, 6.0],
    "alpha2": [-6.0, 6.0],
    "bias_3": [-0.5, 0.5],
    "bias_4": [-0.5, 0.5],
    "bias_ta": [0, 2.0],
}


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.

    """

    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        "Return the locations of the ticks"
        majorlocs = self.axis.get_majorticklocs()
        majorlocs = np.concatenate((majorlocs, np.array([majorlocs[-1] * 10])))

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i - 1]
            if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)
        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError(
            "Cannot get tick locations for a " "%s type." % type(self)
        )


# Hex keys for the color scheme for shear catalog paper
hsc_colors = {
    "GAMA09H": "#d73027",
    "GAMA15H": "#fc8d59",
    "HECTOMAP": "#fee090",
    "VVDS": "#000000",
    "WIDE12H": "#91bfdb",
    "XMM": "#4575b4",
}

hsc_marker = {
    "GAMA09H": "v",
    "GAMA15H": "^",
    "HECTOMAP": "s",
    "VVDS": "p",
    "WIDE12H": "*",
    "XMM": "o",
}

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

colors0 = [
    "black",
    "#1A85FF",
    "#D41159",
    "#DE8817",
    "#A3D68A",
    "#35C3D7",
    "#8B0F8C",
]

cblue = ["#004c6d", "#346888", "#5886a5", "#7aa6c2", "#9dc6e0", "#c1e7ff"]
cred = ["#DC1C13", "#EA4C46", "#F07470", "#F1959B", "#F6BDC0", "#F8D8E3"]


def make_figure_axes(ny=1, nx=1, square=True):
    """Makes figure and axes

    Args:
        ny (int): number of plot in y
        nx (int): number of plot in x
        square (bool): figure is suqare?
    Returns:
        fig (figure): figure
        axes (list): list of axes
    """
    if not isinstance(ny, int):
        raise TypeError("ny should be integer")
    if not isinstance(nx, int):
        raise TypeError("nx should be integer")
    axes = []
    if ny == 1 and nx == 1:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(ny, nx, 1)
        axes.append(ax)
    elif ny == 2 and nx == 1:
        if square:
            fig = plt.figure(figsize=(6, 11))
        else:
            fig = plt.figure(figsize=(6, 7))
        ax = fig.add_subplot(ny, nx, 1)
        axes.append(ax)
        ax = fig.add_subplot(ny, nx, 2)
        axes.append(ax)
    elif ny == 1 and nx == 2:
        fig = plt.figure(figsize=(11, 6))
        for i in range(1, 3):
            ax = fig.add_subplot(ny, nx, i)
            axes.append(ax)
    elif ny == 1 and nx == 3:
        fig = plt.figure(figsize=(18, 6))
        for i in range(1, 4):
            ax = fig.add_subplot(ny, nx, i)
            axes.append(ax)
    elif ny == 1 and nx == 4:
        fig = plt.figure(figsize=(20, 5))
        for i in range(1, 5):
            ax = fig.add_subplot(ny, nx, i)
            axes.append(ax)
    elif ny == 2 and nx == 3:
        fig = plt.figure(figsize=(15, 8))
        for i in range(1, 7):
            ax = fig.add_subplot(ny, nx, i)
            axes.append(ax)
    elif ny == 2 and nx == 4:
        fig = plt.figure(figsize=(20, 8))
        for i in range(1, 9):
            ax = fig.add_subplot(ny, nx, i)
            axes.append(ax)
    else:
        raise ValueError("Do not have option: ny=%s, nx=%s" % (ny, nx))
    return fig, axes


def make_cosebis_ploddt(nmodes):
    """

    Args:
        nmodes (int):   number of cosebis modes
    """
    nzs = 4
    axes = {}
    fig = plt.figure(figsize=(10, 10))

    label1 = r"$E_n [\times 10^{10}]$"
    label2 = r"$B_n [\times 10^{10}]$"
    ll1 = r"$E_n$"
    ll2 = r"$B_n$"

    for i in range(nzs):
        for j in range(i, nzs):
            # -----emode
            ax = plt.subplot2grid(
                (10, 10), ((4 - i) * 2, (3 - j) * 2), colspan=2, rowspan=2, fig=fig
            )
            ax.set_title(
                r"$%d \times %d$" % (i + 1, j + 1), fontsize=15, y=1.0, pad=-15, x=0.8
            )
            ax.grid()

            # x-axis
            ax.set_xlim(0.2, nmodes + 0.8)
            rr = np.arange(1, nmodes + 0.8, 1)
            if i != 0:
                ax.set_xticks(rr)
                ax.set_xticklabels([""] * len(rr))
            else:
                ax.set_xticks(rr)
            if i == 0 and j == 1:
                ax.set_xlabel(r"$n$")
            del rr

            # y-axis
            ax.set_ylim(-5.5 - i * 2, i * 10 + 6.5)
            rr = np.arange(int(-4.5 - i * 2), i * 10 + 5.5, 4 + i * 2)
            if j != nzs - 1:
                ax.set_yticks(rr)
                ax.set_yticklabels([""] * len(rr))
            else:
                ax.set_yticks(rr)
            if j == nzs - 1 and i == 2:
                ax.set_ylabel(label1)
            axes.update({"%d%d_e" % (i + 1, j + 1): ax})
            del rr

            # -----bmode
            ax = plt.subplot2grid(
                (10, 10), (i * 2, (j + 1) * 2), colspan=2, rowspan=2, fig=fig
            )
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_title(
                r"$%d \times %d$" % (i + 1, j + 1), fontsize=15, y=1.0, pad=-15, x=0.8
            )
            ax.grid()

            # x-axis
            ax.set_xlim(0.2, nmodes + 0.8)
            rr = np.arange(1, nmodes + 0.8, 1)
            if i != j:
                ax.set_xticks(rr)
                ax.set_xticklabels([""] * len(rr))
            else:
                ax.set_xticks(rr)

            # y-axis
            ax.set_ylim(-4.5, 4.5)
            rr = np.arange(-4, 4.5, 2)
            if j != nzs - 1:
                ax.set_yticks(rr)
                ax.set_yticklabels([""] * len(rr))
            else:
                ax.set_yticks(rr)
            if i == 2 and j == 3:
                ax.set_ylabel(label2)
            axes.update({"%d%d_b" % (i + 1, j + 1): ax})

    ax = plt.subplot2grid((10, 10), (1 * 2, 1 * 2), colspan=2, rowspan=2, fig=fig)
    ax.set_axis_off()
    leg1 = mlines.Line2D([], [], color=colors[0], marker="+", label=ll1, lw=0)
    leg2 = mlines.Line2D([], [], color=colors[0], marker=".", label=ll2, lw=0)
    ax.legend(handles=[leg1, leg2], loc="lower right", fontsize=20, markerscale=2.0)

    ax = plt.subplot2grid((10, 10), (0, 0), colspan=2, rowspan=2, fig=fig)
    ax.set_axis_off()
    plt.subplots_adjust(wspace=0.12, hspace=0.12)
    return fig, axes


def make_cosebis_bmode_plot(nmodes):
    """

    Args:
        nmodes (int):   number of cosebis modes
    """
    nzs = 4
    axes = {}
    fig = plt.figure(figsize=(8, 8))

    label2 = r"$B_n [\times 10^{10}]$"

    for i in range(nzs):
        for j in range(i, nzs):
            # -----bmode
            ax = plt.subplot2grid((8, 8), (i * 2, j * 2), colspan=2, rowspan=2, fig=fig)
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_title(
                r"$%d \times %d$" % (i + 1, j + 1), fontsize=15, y=1.0, pad=-15, x=0.8
            )
            ax.grid()

            # x-axis
            ax.set_xlim(0.2, nmodes + 0.8)
            rr = np.arange(1, nmodes + 0.8, 1)
            if i != j:
                ax.set_xticks(rr)
                ax.set_xticklabels([""] * len(rr))
            else:
                ax.set_xticks(rr)

            # y-axis
            ax.set_ylim(-4.5, 4.5)
            rr = np.arange(-4, 4.5, 2)
            if j != nzs - 1:
                ax.set_yticks(rr)
                ax.set_yticklabels([""] * len(rr))
            else:
                ax.set_yticks(rr)
            if i == 2 and j == 3:
                ax.set_ylabel(label2)
            axes.update({"%d%d_b" % (i + 1, j + 1): ax})

    ax = plt.subplot2grid((8, 8), (3 * 2, 1 * 2), colspan=2, rowspan=2, fig=fig)
    ax.set_axis_off()
    ll2 = mlines.Line2D(
        [], [], color=colors[0], marker=".", label=r"COSEBIS: $B$-mode", lw=0
    )
    ax.legend(handles=[ll2], loc="lower right", fontsize=20, markerscale=2.0)

    plt.subplots_adjust(wspace=0.12, hspace=0.12)
    return fig, axes


def make_tpcf_plot(
    title="xi", nzs=4, superscript1=None, superscript2=None, small_range=False
):
    """Prepares the frames the two-point correlation corner plot

    Args:
        title (str):    title of the figure ['xi', 'thetaxi', 'ratio', 'ratio2']
        nzs (int):      number of redshift bins
    """
    axes = {}
    fig = plt.figure(figsize=((nzs + 1) * 3, (nzs + 1) * 2))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.xticks([])
    plt.yticks([])

    if superscript1 is None:
        ss1 = r"\xi_{+}"
    elif isinstance(superscript1, str):
        ss1 = r"\xi_{+}^{%s}" % superscript1
    else:
        raise TypeError("superscript1 must be str")
    if superscript2 is None:
        ss2 = r"\xi_{-}"
    elif isinstance(superscript2, str):
        ss2 = r"\xi_{-}^{%s}" % superscript2
    else:
        raise TypeError("superscript2 must be str")

    if title == "xi":
        label1 = r"$%s$" % ss1
        label2 = r"$%s$" % ss2
    elif title == "thetaxi":
        label1 = r"$\theta %s \times 10^4$" % ss1
        label2 = r"$\theta %s \times 10^4$" % ss2
    elif title == "ratio":
        label1 = r"$\delta{%s}/\xi_{+}$" % ss1
        label2 = r"$\delta{%s}/\xi_{-}$" % ss2
    elif title == "ratio2":
        label1 = r"$\delta{%s}/\sigma_{+}$" % ss1
        label2 = r"$\delta{%s}/\sigma_{-}$" % ss2
    else:
        raise ValueError("title should be xi, thetaxi, ratio or ratio2")
    labelsize = 20
    # -----xip---starts
    for i in range(nzs):
        for j in range(i, nzs):
            ax = plt.subplot2grid(
                ((nzs + 1) * 2, (nzs + 1) * 2),
                ((4 - i) * 2, (3 - j) * 2),
                colspan=2,
                rowspan=2,
                fig=fig,
            )
            if title == "thetaxi":
                ax.set_title(
                    r"$%d \times %d$" % (i + 1, j + 1),
                    fontsize=15,
                    y=1.0,
                    pad=-15,
                    x=0.2,
                )
            else:
                ax.set_title(
                    r"$%d \times %d$" % (i + 1, j + 1),
                    fontsize=15,
                    y=1.0,
                    pad=-15,
                    x=0.8,
                )
            # ax.grid()
            # x-axis
            ax.set_xscale("symlog", linthresh=1e-1)
            ax.xaxis.set_minor_locator(MinorSymLogLocator(1e-1))
            if small_range:
                ax.set_xticks([5, 10, 20, 40, 80])
                plt.minorticks_off()
            if i != 0:
                ax.set_xticklabels([])
            else:
                if small_range:
                    ax.set_xticklabels(["5", "10", "20", "40", ""])
            if i == 0 and j == 1:
                ax.set_xlabel(r"$\theta$ [arcmin]")
            # y-axis
            if title == "xi":
                ax.set_ylim(2e-7, 2e-3)
                ax.set_yscale("log")
                if j != nzs - 1:
                    ax.set_yticks((1e-6, 1e-5, 1e-4, 1e-3))
                    ax.set_yticklabels(("", "", "", ""))
                else:
                    ax.set_yticks((1e-6, 1e-5, 1e-4, 1e-3))
                if j == nzs - 1 and i == 2:
                    ax.set_ylabel(label1, fontsize=labelsize)
            elif title == "thetaxi":
                ax.set_ylim(-1, (i + 2) * 2 + 1.8)
                rr = np.arange(0, (i + 2) * 2 + 0.1, 2)
                if j != nzs - 1:
                    ax.set_yticks(rr)
                    ax.set_yticklabels([""] * len(rr))
                else:
                    ax.set_yticks(rr)
                if j == nzs - 1 and i == 2:
                    ax.set_ylabel(label1, fontsize=labelsize)
            elif title in ["ratio", "ratio2"]:
                if j != nzs - 1:
                    ax.set_yticklabels([])
                else:
                    pass
                if j == nzs - 1 and i == 2:
                    ax.set_ylabel(label1, fontsize=labelsize)
            else:
                raise ValueError("title should be xi, thetaxi, ratio or ratio2")
            ax.patch.set_alpha(0.1)
            axes.update({"%d%d_p" % (i + 1, j + 1): ax})
            del ax

    # -----xip---ends

    # -----xim---starts
    for i in range(nzs):
        for j in range(i, nzs):
            ax = plt.subplot2grid(
                ((nzs + 1) * 2, (nzs + 1) * 2),
                (i * 2, (j + 1) * 2),
                colspan=2,
                rowspan=2,
                fig=fig,
            )
            if title == "thetaxi":
                ax.set_title(
                    r"$%d \times %d$" % (i + 1, j + 1),
                    fontsize=15,
                    y=1.0,
                    pad=-15,
                    x=0.2,
                )
            else:
                ax.set_title(
                    r"$%d \times %d$" % (i + 1, j + 1),
                    fontsize=15,
                    y=1.0,
                    pad=-15,
                    x=0.8,
                )
            # ax.grid()
            # x-axis
            ax.set_xscale("symlog", linthresh=1e-1)
            ax.xaxis.set_minor_locator(MinorSymLogLocator(1e-1))

            if small_range:
                ax.set_xticks([50, 100, 200])
                plt.minorticks_off()
            if i != j:
                ax.set_xticklabels([])
            else:
                if small_range:
                    ax.set_xticklabels(["50", "100", "200"])

            # y-axis
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            if title == "xi":
                ax.set_ylim(1e-7, 1e-3)
                ax.set_yscale("log")
                if j != nzs - 1:
                    ax.set_yticks((1e-6, 1e-5, 1e-4, 1e-3))
                    ax.set_yticklabels(("", "", "", ""))
                else:
                    ax.set_yticks((1e-6, 1e-5, 1e-4, 1e-3))
                if i == 2 and j == 3:
                    ax.set_ylabel(label2, fontsize=labelsize)
            elif title == "thetaxi":
                ax.set_ylim(-1, (i + 1) * 2 + 1.8)
                rr = np.arange(0, (i + 1) * 2 + 0.1, 2)
                if j != nzs - 1:
                    ax.set_yticks(rr)
                    ax.set_yticklabels([""] * len(rr))
                else:
                    ax.set_yticks(rr)
                if i == 2 and j == 3:
                    ax.set_ylabel(label2, fontsize=labelsize)
            elif title in ["ratio", "ratio2"]:
                if j != nzs - 1:
                    ax.set_yticklabels([])
                else:
                    pass
                if i == 2 and j == 3:
                    ax.set_ylabel(label2, fontsize=labelsize)
            else:
                raise ValueError("title should be xi, thetaxi or ratio")
            ax.patch.set_alpha(0.1)
            axes.update({"%d%d_m" % (i + 1, j + 1): ax})
            del ax
    # -----xim---ends
    return fig, axes


def plot_cov_coeff(covIn):
    """Makes plot for covariance matrix for cosmic shear

    Args:
        covIn (ndarray): covariance coefficients
    Returns:
        fig (figure):   figure for covariance coefficients
    """
    fig, axes = make_figure_axes(nx=1, ny=1)
    ax = axes[0]
    im = ax.imshow(
        covIn, cmap="coolwarm", vmin=-1, vmax=1, origin="lower", aspect="auto"
    )
    fig.colorbar(im)
    ny, nx = covIn.shape
    ax.set_xticks(np.arange(0, nx + 1, 30))
    ax.set_yticks(np.arange(0, ny + 1, 30))
    return fig


def plot_chain_corner(
    clist,
    cnlist,
    blind_by,
    nlist,
    truth=None,
    scale=2.5,
    stat_method=stat0,
    kde=kde0,
    plot_hists=True,
    shade=False,
    color_use=None,
    line_width=1.5,
    line_styles=None,
    ax=None,
    contour_labels=None,
):
    """Makes the corner plots for posteriors

    Args:
        clist (list):       a list of MC chain with nested sampling
        cnlist (list):      a list of chain names
        blind_by (str):     whether to blind_by the reaults
        nlist (list):       a list of parameters
        truth (list):       a list of truth parameters
        kde (float):        kernal density esitamte
        plot_hists (bool):  whether ploting 1D histogram
        shade (bool):       whether shade or not
        color_use (list):   colors for contours
        line_width (float): line width
        contour_labels (float):
                            contour label, "sigma" or "confidence"
    Returns:
        fig (figure):       figure
    """
    if scale <= 1.1:
        sigmas = [0.0, 0.3, 0.5]
    elif scale < 2.0:
        sigmas = [0, 1]
    elif scale < 3:
        sigmas = [0, 1, 2]
    elif scale < 10:
        sigmas = [0, 1, 2, 3]
    else:
        raise ValueError("scale need to be greater than 1.05 and smaller than 10")
    if plot_hists:
        loc = "lower center"
    else:
        loc = "upper right"

    if line_styles is None:
        line_styles = ["-"] * len(clist)
    elif isinstance(line_styles, str):
        line_styles = [line_styles] * len(clist)

    if line_width is None:
        line_width = [1.0] * len(clist)
    elif isinstance(line_width, str):
        line_width = [line_width] * len(clist)

    if shade is None:
        shade = [False] * len(clist)
    elif isinstance(shade, bool):
        shade = [shade] * len(clist)

    npar = len(nlist)
    if blind_by is not None:
        assert (
            blind_by in cnlist
        ), "The blinding prarmeter ('blind_by') \
            is not in the chain name list (cnlist)"
        c = ChainConsumer()
        oo = clist[cnlist.index(blind_by)]
        c.add_chain(
            [oo[ni] for ni in nlistb],
            weights=oo["weight"],
            parameters=[latexDict[nn] for nn in nlistb],
            posterior=oo["post"],
            kde=kde,
            statistics=stat_method,
            plot_point=False,
        )
        stat = c.analysis.get_summary()
        avel = np.array([stat[latexDict[ni]][1] for ni in nlistb])
        del c, oo, stat
    else:
        avel = np.array([0] * len(nlistb))

    c = ChainConsumer()
    if color_use is not None:
        assert len(color_use) == len(clist)
    for ii, oo in enumerate(clist):
        # blind sigma_8 and omega_m
        chain_name = cnlist[ii]
        nlist2 = [nn for nn in nlist if nn in oo.dtype.names]
        ll = [oo[nn] - avel[nlistb.index(nn)] for nn in nlist2 if nn in nlistb]
        ll2 = [oo[nn] for nn in nlist2 if nn not in nlistb]
        if color_use is None:
            c.add_chain(
                ll + ll2,
                weights=oo["weight"],
                parameters=[latexDict[nn] for nn in nlist2],
                posterior=oo["post"],
                kde=kde,
                name=chain_name,
                statistics=stat_method,
                plot_point=False,
            )
        else:
            c.add_chain(
                ll + ll2,
                weights=oo["weight"],
                parameters=[latexDict[nn] for nn in nlist2],
                posterior=oo["post"],
                kde=kde,
                name=chain_name,
                statistics=stat_method,
                plot_point=False,
                color=color_use[ii],
            )
        del ll, ll2, nlist2
    if len(cnlist) <= 3:
        ncol = 1
        fw = "bold"
    elif len(cnlist) <= 5:
        ncol = 1
        fw = "bold"
    elif len(cnlist) <= 10:
        ncol = 2
        fw = "bold"
    else:
        ncol = 2
        fw = "bold"

    if len(nlist) <= 2:
        fontsize = 28
        bb = 0.2
        tt = 0.2
    elif len(nlist) < 5:
        fontsize = 30
        bb = 0.12
        tt = 0.12
    elif len(nlist) < 10:
        fontsize = 32
        bb = 0.1
        tt = 0.1
    else:
        fontsize = 35
        bb = 0.05
        tt = 0.05
    c.configure(
        serif=True,
        usetex=True,
        global_point=False,
        shade=shade,
        shade_alpha=0.70,
        plot_hists=plot_hists,
        flip=False,
        bar_shade=True,
        statistics=stat_method,
        label_font_size=int(fontsize / 1.0) + 1,
        tick_font_size=int(fontsize / 1.2) + 1,
        linewidths=line_width,
        linestyles=line_styles,
        spacing=0.0,
        max_ticks=3,
        sigmas=sigmas,
        summary=True,
        norm_max=True,
        legend_artists=True,
        legend_kwargs={
            "loc": loc,
            "prop": {
                "weight": fw,
                "size": int(fontsize - len(cnlist) * 2.0 + 4),
            },
            "ncol": ncol,
            "columnspacing": 0.2,
        },
        contour_labels=contour_labels,
        contour_label_font_size=15,
    )
    stat = np.atleast_1d(c.analysis.get_summary())
    lnlist = [latexDict[nn] for nn in nlist]
    idx = np.sort(np.unique(lnlist, return_index=True)[1])
    nlist2 = [nlist[ii] for ii in idx]
    if ax is None:
        exts = get_summary_extents(stat, nlist2, clist, scale=scale, blind_shift=avel)
        fig = c.plotter.plot(figsize=2.0, extents=exts, truth=truth)
        fig.subplots_adjust(bottom=bb, left=tt)
        c.plotter.restore_rc_params()
        return fig
    else:
        assert len(nlist) == 2
        c.plotter.plot_contour(ax, latexDict[nlist[0]], latexDict[nlist[1]])
        c.plotter.restore_rc_params()
        return


def get_summary_extents(stat, pnlist, clist, scale=1.0, blind_shift=None):
    """Estimates the extent for chains used for summary plot

    Args:
        stat (statistics):  statistics of chainconsumer
        pnlist (list):      a list of parameter names
        clist (list):       chain list [a list of ndarray]
    Returns:
        ext (list):         a list of tuples of extents for parameters
        scale (float):      scale ratio
    """
    if blind_shift is None:
        blind_shift = np.zeros(len(nlistb))
    ldt = latexDict
    npar = len(pnlist)
    nchain = len(clist)
    emin = []
    emax = []
    for j in range(nchain):
        min_tmp = []
        max_tmp = []
        stat_tmp = stat[j]
        for i in range(npar):
            pname = ldt[pnlist[i]]
            if pname in stat_tmp.keys():
                min_tmp.append(stat_tmp[pname][0])
                max_tmp.append(stat_tmp[pname][-1])
            else:
                min_tmp.append(np.nan)
                max_tmp.append(np.nan)
        emin.append(min_tmp)
        emax.append(max_tmp)

    emin = np.nanmin(np.array(emin), axis=0)
    emax = np.nanmax(np.array(emax), axis=0)
    ecen = (emin + emax) / 2.0
    edd = np.max(np.stack([np.abs(emin - ecen), np.abs(emax - ecen)]), axis=0) * 1.32
    exts = [[ecen[i] - edd[i] * scale, ecen[i] + edd[i] * scale] for i in range(npar)]
    for ie, ee in enumerate(exts):
        nn = pnlist[ie]
        if nn in rangeDict.keys():
            if nn in nlistb:
                low = rangeDict[nn][0] - blind_shift[nlistb.index(nn)]
                high = rangeDict[nn][1] - blind_shift[nlistb.index(nn)]
            else:
                low = rangeDict[nn][0]
                high = rangeDict[nn][1]
            ee[0] = max(low, ee[0])
            ee[1] = min(high, ee[1])
    return exts


def get_summary_lims(stat, pnlist, clist):
    """Estimates the extent for chains used for summary plot

    Args:
        stat (statistics):  statistics of chainconsumer
        pnlist (list):      a list of parameter names
        clist (list):       chain list [a list of ndarray]
    Returns:
        parlims (list):     a list of tuples of extents for parameters
    """
    npar = len(pnlist)
    nchain = len(clist)
    parlims = [
        [stat[j][latexDict[pnlist[i]]] for i in range(npar)] for j in range(nchain)
    ]
    return parlims


def plot_chain_summary(
    clist,
    cnlist,
    blind_by="fid",
    pnlist=None,
    nstat=1,
    stat_method=stat0,
    kde=kde0,
):
    """Plots the summary for a list of chains

    Args:
        clist (list):       chain list [a list of ndarray]
        cnlist (list):      chain name list [a list of str]
        blind_by (str):     whether to blind_by the reaults
        pnlist (list):      parameter list
        nstat (int):        number of statistics [MEAN and MAP]
        kde (float):        kernal density esitamte
    Return:
        fig (figure):       mpl figure
    """
    fmts = ["o", "d"]
    alphas = [1.0, 1.0]
    if pnlist is None:
        pnlist = ["omega_m", "sigma_8", "s_8"]
    npar = len(pnlist)
    nchain = len(clist)

    # blinding
    if blind_by is not None:
        assert (
            blind_by in cnlist
        ), "The blinding prarmeter ('blind_by') \
            is not in the chain name list (cnlist)"
        c = ChainConsumer()
        oo = clist[cnlist.index(blind_by)]
        c.add_chain(
            [oo[ni] for ni in nlistb],
            weights=oo["weight"],
            parameters=[latexDict[nn] for nn in nlistb],
            posterior=oo["post"],
            kde=kde,
            statistics=stat_method,
            plot_point=False,
        )
        stat = c.analysis.get_summary()
        avel = np.array([stat[latexDict[ni]][1] for ni in nlistb])
        del c, oo, stat
    else:
        avel = np.array([0] * len(nlistb))

    c = ChainConsumer()
    for ii, oo in enumerate(clist):
        # blind sigma_8 and omega_m
        chain_name = cnlist[ii]
        nlist2 = [nn for nn in pnlist if nn in oo.dtype.names]
        ll = [oo[nn] - avel[nlistb.index(nn)] for nn in nlist2 if nn in nlistb]
        ll2 = [oo[nn] for nn in nlist2 if nn not in nlistb]
        c.add_chain(
            ll + ll2,
            weights=oo["weight"],
            parameters=[latexDict[nn] for nn in nlist2],
            posterior=oo["post"],
            kde=kde,
            name=chain_name,
            statistics=stat_method,
            plot_point=False,
        )
        del ll, ll2, nlist2

    c.configure(global_point=False, statistics=stat_method)
    stat = np.atleast_1d(c.analysis.get_summary())
    parlims = get_summary_lims(stat, pnlist, clist)
    extent = get_summary_extents(stat, pnlist, clist, scale=1.5)

    if nstat == 1:
        parlims = [parlims]
    elif nstat == 2:
        pass
        # c.configure(global_point=True, statistics=stat_method)
        # stat = c.analysis.get_summary()
        # parlims2 = get_summary_lims(stat, pnlist, clist)
        # parlims = [parlims, parlims2]
        # del stat, parlims2
    else:
        raise ValueError("nstat can only be 1 or 2.")

    fig, axes = plt.subplots(
        1,
        npar,
        sharey=True,
        figsize=((npar + 1) * 2, nchain // 2 + 1),
    )
    lower0 = None
    upper0 = None
    for h in range(nstat):
        # iteration on stats
        for j in range(npar):
            # iteration on parameters
            for i in range(nchain):
                # iteration on chains
                lower, mid, upper = tuple(parlims[h][i][j])
                if h == 0 and i == 0:
                    lower0 = lower
                    upper0 = upper
                err = np.array([[mid - lower, upper - mid]]).reshape(2, 1)
                axes[j].errorbar(
                    [mid],
                    [i + h * 0.3],
                    xerr=err,
                    fmt=fmts[h],
                    alpha=alphas[h],
                    elinewidth=1.6,
                    color=colors[0],
                )
            if h == 0:
                axes[j].tick_params(axis="both", which="major", labelsize=16)
                axes[j].axvspan(lower0, upper0, color="gray", alpha=0.2)
                axes[j].set_title(latexDict[pnlist[j]], fontsize=20)
                axes[j].set_xlim(extent[j])
    axes[0].tick_params(axis="y", which="major", pad=10)
    plt.ylim(-0.5, nchain - 0.5)
    plt.yticks(range(nchain), cnlist)
    plt.gca().invert_yaxis()
    plt.subplots_adjust(
        wspace=0.12,
        hspace=0.0,
        top=0.8,
        bottom=0.2,
        left=0.20,
    )
    return fig


def plot_pvalue_list(plist, nlist):
    """Makes plots of p values with different chains

    Args:
        plist (list):   a list of p-values
        nlist (list):   a list of chain names
    Return
        fig (Figure):   matplotlib figure
    """
    if not isinstance(plist, list):
        raise TypeError("plist should be a list")
    if not isinstance(nlist, list):
        raise TypeError("nlist should be a list")
    if not len(plist) == len(nlist):
        raise ValueError("plist and nlist should have same length")
    for pp in plist:
        assert isinstance(pp, (float, int))
    for nn in nlist:
        assert isinstance(nn, str)

    nchain = len(plist)
    fig, axes = make_figure_axes(1, 1)
    ax = axes[0]
    xarray = np.arange(nchain)
    ax.plot(xarray, np.array(plist), marker="o", color=colors0[1])
    ax.set_xticks(xarray, nlist)
    ax.set_ylabel("$p$-value")
    ax.grid()
    return fig


def plot_xipm_data(fname, axes, marker="x", color=colors0[0], nzs=4, extnms=None):
    """Makes corner plots for xip and xim from cosmosis data file [fits]

    Args:
        fname (str):    a fits file name
        axes (dict):    a dictionary of axis generated by `make_tpcf_plot`
        marker (str):   marker in plot
        color (str):    color of marker and error bar
        nzs (int):      number of redshift bins
    """
    hdul = pyfits.open(fname)
    if extnms is None:
        extnms = ["xi_plus", "xi_minus"]
    cov2 = hdul["COVMAT"].data
    err0 = np.sqrt(np.diag(cov2))
    np0 = hdul["COVMAT"].header["STRT_0"]
    nm0 = hdul["COVMAT"].header["STRT_1"]
    nzall = int(nzs * (nzs + 1) // 2)
    npd = (nm0) // nzall
    nmd = (cov2.shape[0] - nm0) // nzall
    err_xip = err0[np0:nm0]
    err_xip = err_xip.reshape(err_xip.size // npd, npd)
    err_xim = err0[nm0:]
    err_xim = err_xim.reshape(err_xim.size // nmd, nmd)
    ic = 0
    for i in range(nzs):
        for j in range(i, nzs):
            ax = axes["%d%d_p" % (i + 1, j + 1)]
            msk = (hdul[extnms[0]].data["BIN1"] == (i + 1)) & (
                hdul[extnms[0]].data["BIN2"] == (j + 1)
            )
            dd = hdul[extnms[0]].data[msk]
            xx = dd["ang"]
            yy = dd["value"] * xx * 1e4
            yerr = err_xip[ic] * xx * 1e4
            ax.errorbar(
                xx,
                yy,
                yerr,
                marker=marker,
                linestyle="",
                color=color,
                markersize=4.0,
                markeredgewidth=2.0,
            )
            del xx, yy, msk, dd
            # ---
            ax = axes["%d%d_m" % (i + 1, j + 1)]
            msk = (hdul[extnms[1]].data["BIN1"] == (i + 1)) & (
                hdul[extnms[1]].data["BIN2"] == (j + 1)
            )
            dd = hdul[extnms[1]].data[msk]
            xx = dd["ang"]
            yy = dd["value"] * xx * 1e4
            yerr = err_xim[ic] * xx * 1e4
            ax.errorbar(
                xx,
                yy,
                yerr,
                marker=marker,
                linestyle="",
                color=color,
                markersize=4.0,
                markeredgewidth=2.0,
            )
            del xx, yy, dd
            ic += 1
    return


def plot_xipm_error(fname, axes, marker="x", color=colors0[0], nzs=4, extnms=None):
    """Makes corner plots for xip and xim from cosmosis data file [fits]

    Args:
        fname (str):    a fits file name
        axes (dict):    a dictionary of axis generated by `make_tpcf_plot`
        marker (str):   marker in plot
        color (str):    color of marker and error bar
        nzs (int):      number of redshift bins
    """
    hdul = pyfits.open(fname)
    if extnms is None:
        extnms = ["xi_plus", "xi_minus"]
    cov2 = hdul["COVMAT"].data
    err0 = np.sqrt(np.diag(cov2))
    np0 = hdul["COVMAT"].header["STRT_0"]
    nm0 = hdul["COVMAT"].header["STRT_1"]
    nzall = int(nzs * (nzs + 1) // 2)
    npd = (nm0) // nzall
    nmd = (cov2.shape[0] - nm0) // nzall
    err_xip = err0[np0:nm0]
    err_xip = err_xip.reshape(err_xip.size // npd, npd)
    err_xim = err0[nm0:]
    err_xim = err_xim.reshape(err_xim.size // nmd, nmd)
    ic = 0
    for i in range(nzs):
        for j in range(i, nzs):
            ax = axes["%d%d_p" % (i + 1, j + 1)]
            msk = (hdul[extnms[0]].data["BIN1"] == (i + 1)) & (
                hdul[extnms[0]].data["BIN2"] == (j + 1)
            )
            dd = hdul[extnms[0]].data[msk]
            xx = dd["ang"]
            yerr = err_xip[ic] * xx * 1e4
            ax.plot(
                xx,
                yerr,
                marker=marker,
                linestyle="",
                color=color,
                markersize=4.0,
            )
            del xx, yerr, msk, dd
            # ---
            ax = axes["%d%d_m" % (i + 1, j + 1)]
            msk = (hdul[extnms[1]].data["BIN1"] == (i + 1)) & (
                hdul[extnms[1]].data["BIN2"] == (j + 1)
            )
            dd = hdul[extnms[1]].data[msk]
            xx = dd["ang"]
            yerr = err_xim[ic] * xx * 1e4
            ax.errorbar(
                xx,
                yerr,
                marker=marker,
                linestyle="",
                color=color,
                markersize=4.0,
            )
            del xx, yerr, dd
            ic += 1
    return


def plot_xipm_model(
    Dir,
    axes,
    ls="-",
    color=colors0[0],
    nzs=4,
    blind=False,
    Dir2=None,
    pmin=5,
    pmax=80,
    mmin=25,
    mmax=300,
):
    """Makes cornor plots for xip and xim from cosmosis output files [model
    prediction]

    Args:
        Dir (str):      output directory name of cosmosis test
        axes (dict):    a dictionary of axis generated by `make_tpcf_plot`
        ls (str):       line style in plot
        color (str):    color of marker and error bar
        nzs (int):      number of redshift bins
        blind (bool):   whether do blinding
        Dir2 (str):     output directory name of cosmosis test [be subtracted]
    """
    infname = os.path.join(Dir, "shear_xi_plus/theta.txt")
    thetaP = np.loadtxt(infname) / np.pi * 180.0 * 60.0

    infname = os.path.join(Dir, "shear_xi_minus/theta.txt")
    thetaM = np.loadtxt(infname) / np.pi * 180.0 * 60.0
    del infname
    for i in range(nzs):
        for j in range(i, nzs):
            ax = axes["%d%d_p" % (i + 1, j + 1)]
            xx = thetaP
            yy = (
                np.loadtxt(
                    os.path.join(Dir, "shear_xi_plus/bin_%d_%d.txt" % (j + 1, i + 1))
                )
                * xx
                * 1e4
            )
            if Dir2 is not None:
                yy = yy - (
                    np.loadtxt(
                        os.path.join(
                            Dir2, "shear_xi_plus/bin_%d_%d.txt" % (j + 1, i + 1)
                        )
                    )
                    * xx
                    * 1e4
                )
            ax.plot(xx, yy, linestyle=ls, color=color, linewidth=2.0)
            ax.set_xlim(pmin, pmax)
            if blind:
                ax.set_yticks([])
            ax.axvspan(1, 7.13, color="gray", alpha=0.4)
            ax.axvspan(56.52, 100, color="gray", alpha=0.4)
            del ax, xx, yy

            ax = axes["%d%d_m" % (i + 1, j + 1)]
            xx = thetaM
            yy = (
                np.loadtxt(
                    os.path.join(Dir, "shear_xi_minus/bin_%d_%d.txt" % (j + 1, i + 1))
                )
                * xx
                * 1e4
            )
            if Dir2 is not None:
                yy = yy - (
                    np.loadtxt(
                        os.path.join(
                            Dir2, "shear_xi_minus/bin_%d_%d.txt" % (j + 1, i + 1)
                        )
                    )
                    * xx
                    * 1e4
                )
            ax.plot(xx, yy, linestyle=ls, color=color, linewidth=2.0)
            ax.set_xlim(mmin, mmax)
            if blind:
                ax.set_yticks([])
            ax.axvspan(1, 31.2, color="gray", alpha=0.4)
            ax.axvspan(248.0, 500, color="gray", alpha=0.4)
            del ax, xx, yy
    return


def plot_xipm_data_model_ratio(
    fname,
    Dir,
    axes,
    ls="-",
    marker=".",
    color=colors0[0],
    nzs=4,
    blind=False,
    pmin=5,
    pmax=80,
    mmin=25,
    mmax=300,
    extnms=None,
):
    """Makes cornor plots for xip and xim from cosmosis output files [model
    prediction]

    Args:
        Dir (str):      output directory name of cosmosis test
        axes (dict):    a dictionary of axis generated by `make_tpcf_plot`
        ls (str):       line style in plot
        color (str):    color of marker and error bar
        nzs (int):      number of redshift bins
        blind (bool):   whether do blinding
    """

    # read data
    hdul = pyfits.open(fname)
    if extnms is None:
        extnms = ["xi_plus", "xi_minus"]
    cov2 = hdul["COVMAT"].data
    err0 = np.sqrt(np.diag(cov2))
    np0 = hdul["COVMAT"].header["STRT_0"]
    nm0 = hdul["COVMAT"].header["STRT_1"]
    nzall = int(nzs * (nzs + 1) // 2)
    npd = (nm0) // nzall
    nmd = (cov2.shape[0] - nm0) // nzall
    err_xip = err0[np0:nm0]
    err_xip = err_xip.reshape(err_xip.size // npd, npd)
    err_xim = err0[nm0:]
    err_xim = err_xim.reshape(err_xim.size // nmd, nmd)

    # for model
    infname = os.path.join(Dir, "shear_xi_plus/theta.txt")
    thetaP = np.loadtxt(infname) / np.pi * 180.0 * 60.0
    infname = os.path.join(Dir, "shear_xi_minus/theta.txt")
    thetaM = np.loadtxt(infname) / np.pi * 180.0 * 60.0
    del infname

    ic = 0
    for i in range(nzs):
        for j in range(i, nzs):
            ax = axes["%d%d_p" % (i + 1, j + 1)]
            xx = thetaP
            yy = np.loadtxt(
                os.path.join(Dir, "shear_xi_plus/bin_%d_%d.txt" % (j + 1, i + 1))
            )
            mod = Interp1d(xx, yy)
            del xx, yy

            msk = (hdul[extnms[0]].data["BIN1"] == (i + 1)) & (
                hdul[extnms[0]].data["BIN2"] == (j + 1)
            )
            dd = hdul[extnms[0]].data[msk]
            xx = dd["ang"]
            yy = dd["value"]
            yerr = err_xip[ic]
            ymod = mod(xx)
            ax.errorbar(
                xx,
                (yy - ymod) / yerr,
                1,
                marker=marker,
                linestyle=ls,
                color=color,
                linewidth=1.0,
            )

            ax.set_xlim(pmin, pmax)
            ax.set_ylim(-5.8, 5.8)
            ax.axvspan(7.13, 56.52, color=colors[-1], alpha=0.4)
            if blind:
                ax.set_yticks([])
            del xx, yy, ymod, msk, dd, ax, mod

            ax = axes["%d%d_m" % (i + 1, j + 1)]
            xx = thetaM
            yy = np.loadtxt(
                os.path.join(Dir, "shear_xi_minus/bin_%d_%d.txt" % (j + 1, i + 1))
            )
            mod = Interp1d(xx, yy)
            del xx, yy

            msk = (hdul[extnms[1]].data["BIN1"] == (i + 1)) & (
                hdul[extnms[1]].data["BIN2"] == (j + 1)
            )
            dd = hdul[extnms[1]].data[msk]
            xx = dd["ang"]
            yy = dd["value"]
            yerr = err_xim[ic]
            ymod = mod(xx)
            ax.errorbar(
                xx,
                (yy - ymod) / yerr,
                1,
                marker=marker,
                linestyle="",
                color=color,
                linewidth=1.0,
            )
            ax.set_xlim(mmin, mmax)
            ax.set_ylim(-5.8, 5.8)
            ax.axvspan(31.2, 247.0, color=colors[-1], alpha=0.4)
            if blind:
                ax.set_yticks([])
            del ax, xx, yy, ymod, dd, mod, msk
            ic += 1
    return


def nestcheck_plot(infname, n_simulate=100, blind=False, s8_only=False):
    """Plots nestcheck

    Args:
        infname (str):      input file name
        n_simulate (int):   number of simulations
        blind (bool):       whether blind values
        s8_only (bool):     whether only plot s8
    Returns:
        fig (figure):       matplotlib figure
    """
    output_info = TextColumnOutput.load_from_options({"filename": infname})
    colnames, data, metadata, _, final_meta = output_info
    names_all = [nn.split("--")[-1].lower() for nn in colnames]

    def get_id(name):
        return np.where(np.array(names_all) == name)[0][0]

    file_root = metadata[0]["polychord_outfile_root"]
    base_dir = os.path.join(metadata[0]["workdir"], metadata[0]["base_dir"])
    run = data_processing.process_polychord_run(file_root, base_dir)
    # Nest check
    if not s8_only:
        names = ["omega_m", "sigma_8", "s_8"]
        fthetas = [eval("lambda x: x[:,%d]" % get_id(name)) for name in names]
        labels = [latexDict[nn] for nn in names]
        fig = plots.param_logx_diagram(
            run,
            fthetas=fthetas,
            ftheta_lims=[[0.05, 0.5], [0.2, 1.2], [0.4, 1.0]],
            logx_min=-30,
            labels=labels,
            n_simulate=n_simulate,
            colors=cblue,
            colormaps=["Blues_r"],
        )
        if blind:
            for ii in [2, 4, 6]:
                fig.axes[ii].set_yticklabels([])
    else:
        names = ["s_8"]
        fthetas = [eval("lambda x: x[:,%d]" % get_id(name)) for name in names]
        labels = [latexDict[nn] for nn in names]
        fig = plots.param_logx_diagram(
            run,
            fthetas=fthetas,
            ftheta_lims=[[0.4, 1.0]],
            logx_min=-30,
            labels=labels,
            n_simulate=n_simulate,
            colors=cblue,
            colormaps=["Blues_r"],
        )
        if blind:
            for ii in [2]:
                fig.axes[ii].set_yticklabels([])
    return fig
