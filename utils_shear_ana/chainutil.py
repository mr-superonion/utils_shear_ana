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

import numpy as np
from scipy import stats
from chainconsumer import ChainConsumer

from cosmosis.output.text_output import TextColumnOutput

wmap9Dict = {
    "omega_m": 0.279,
    "omega_b": 0.046,
    "sigma_8": 0.82,
    "s_8": 0.79076,
    "a_s": 2.1841e-9,
    "n_s": 0.97,
    "h0": 0.7,
}


def pvalue_of_chi2(chi2, dof, hartlap_nsim=1404, hartlap_ndata=140):
    """Estimates pvalue from chi2 and degree of freedom. Note, if you use
    sellentin correction in cosmosis, the returned 2pt_chi2 need to be
    corrected with hartlap factor.

    Args:
        chi2 (float):           chi2
        dof (float):            effective degree of freedom
        hartlap_nsim (int):     number of simulation for covariance estimation
        hartlap_ndata (int):    number of data points
    Returns:
        p(float):       p-value
    """
    if hartlap_nsim > 0:
        corr_chi2 = (hartlap_nsim - hartlap_ndata - 2) / (hartlap_nsim - 1)
        chi2 = chi2 * corr_chi2
    p = 1.0 - stats.chi2.cdf(chi2, round(dof))
    return p


def read_cosmosis_chain(infname, flip_dz=True, as_correction=True):
    """Reads the chain output of cosmosis into ndarray

    Args:
        infname (str):          file name
    Returns:
        out (ndarray):          chain
    """
    # use cosmosis module to read the output
    try:
        output_info = TextColumnOutput.load_from_options({"filename": infname})
    except IOError:
        raise IOError("Cannot read file: %s" % infname)
    colnames, data, metadata, _, final_meta = output_info
    metadata = metadata[0]
    final_meta = final_meta[0]
    metadata.update(final_meta)

    # data
    data = data[0]
    nsample, npar = data.shape
    # assert nsample==metadata['nsample']

    # initialize dtypes
    # column names
    colnames = [c.lower().split("--")[-1] for c in colnames]
    types = list(set([tp for tp in zip(colnames, [">f8"] * npar)]))
    cal_s8 = False
    if ("s_8" not in colnames) and ("omega_m" in colnames) and ("sigma_8" in colnames):
        types.append(("s_8", ">f8"))
        cal_s8 = True
    if "weight" not in colnames:
        types.append(("weight", ">f8"))
        has_weight = False
    else:
        has_weight = True

    # initialize ndarray
    out = np.empty(nsample, dtype=types)
    for nm in colnames:
        out[nm] = data.T[colnames.index(nm)]
    if cal_s8:
        alpha = 0.5
        out["s_8"] = out["sigma_8"] * (out["omega_m"] / 0.3) ** alpha

    # pre processing
    if has_weight:
        out = out[out["weight"] > 0]
    else:
        out["weight"] = 1.
    # flip the delta_z
    if flip_dz:
        for i in range(1, 20):
            inm = "bias_%d" % i
            if inm in colnames:
                out[inm] = -1 * out[inm]
            else:
                break
    if as_correction:
        if "lnAs" in infname:
            out["weight"] = out["weight"] * out["sigma_8"]
        else:
            out["weight"] = out["weight"] * out["sigma_8"] / out["a_s"]
    return out


def read_cosmosis_max(infname):
    """Reads the maxlike output of cosmosis

    Args:
        infname (str):      file name
    Returns:
        out (ndarray):      chain
    """
    output_info = TextColumnOutput.load_from_options({"filename": infname})
    colnames, data, _, _, _ = output_info
    data = data[0]
    colnames = [c.lower().split("--")[-1] for c in colnames]
    nsample, npar = data.shape
    types = list(set([tp for tp in zip(colnames, [">f8"] * npar)]))
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


def estimate_parameters_from_chain(infname, ptype="map", do_write=True):
    """Estimate the parameters and write it to file

    Args:
        infname (str):      input file name for chain
        chain (ndarray):    input chain [default: None]
        do_write (bool):    whether write down the estimated parameters
    Returns:
        max_post (dict):    maximum a posterior
    """
    chain = read_cosmosis_chain(infname, flip_dz=False, as_correction=True)

    outfname = infname[:-4] + "_%s.ini" % ptype
    names = list(chain.dtype.names)

    # remove those derived parameters
    if ptype != "map":
        for nn in ["prior", "like", "post", "weight"]:
            names.remove(nn)

    c = ChainConsumer()
    c.add_chain(
        [chain[nn] for nn in names],
        weights=chain["weight"],
        parameters=names,
        posterior=chain["post"],
    )
    if ptype == "map":
        max_post = c.analysis.get_max_posteriors()
    else:
        c.configure(
            global_point=True,
            statistics=ptype,
        )
        return
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
                        tmp = "%s = %.6E\n" % (nn, max_post[nn])
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


# from getdist import MCSamples
# from tensiometer import gaussian_tension
# def get_neff_from_chain(data):
#     """Gets the effective degrees of freedom from nested chain

#     Args:
#         data (ndarray):  nested chains
#     Returns:
#         neff (float):  effective degrees of freedom
#     """
#     names = list(data.dtype.names)
#     for nn in names:
#         if nn in ["like", "post", "weight", "prior", "s_8"]:
#             names.remove(nn)
#     c = MCSamples(
#         samples=[data[nn] for nn in names],
#         loglikes=data["like"],
#         weights=data["weight"],
#         names=names,
#     )
#     p = MCSamples(
#         samples=[data[nn] for nn in names], names=names, weights=np.exp(data["prior"])
#     )
#     neff = gaussian_tension.get_Neff(c, prior_chain=p)
#     return neff
