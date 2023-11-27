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
        flip_dz (bool):         whether to flip sign of dz in cosmosis
        as_correction (bool):   whether to add correction factor for sampling
                                on A_s or lnAs
    Returns:
        out (ndarray):          an array of chain
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

    if ("a_s" not in colnames) and ("ln_as1e10" in colnames):
        types.append(("a_s", ">f8"))
    # initialize ndarray
    out = np.zeros(nsample, dtype=types)
    for nm in colnames:
        out[nm] = data.T[colnames.index(nm)]

    # some missing parameters
    if cal_s8:
        alpha = 0.5
        out["s_8"] = out["sigma_8"] * (out["omega_m"] / 0.3) ** alpha
    if ("a_s" not in colnames) and ("ln_as1e10" in colnames):
        out["a_s"] = np.exp(out["ln_as1e10"]) * 1e-10

    # pre processing
    if has_weight:
        out = out[out["weight"] > 0]
    else:
        out["weight"] = 1.0

    # a_s has unit of [1e-9]
    if "a_s" in colnames:
        out["a_s"] = out["a_s"] * 1e9
    if "ombh2" in colnames:
        out["ombh2"] = out["ombh2"] * 1e3

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


def estimate_parameters_from_chain(
    infname, blind=True, do_write=True, params_dict=None
):
    """Estimate the parameters and write it to file

    Args:
        infname (str):      input file name for chain
        blind (bool):       whether blind the output
        do_write (bool):    whether write down the estimated parameters
        params_dict (dict): A dictionary of parameters to replace
    Returns:
        max_post (dict):    maximum a posterior
    """
    ptype = "map"
    chain = read_cosmosis_chain(infname, flip_dz=False, as_correction=True)
    # change the parameters in the param_dict
    if params_dict is not None:
        for par in params_dict.keys():
            chain[par] = params_dict[par]
        pname = "changed"
    else:
        pname = "estimated"
    if blind:
        # use wmap9 cosmology to blind the output
        chain["omega_m"] = wmap9Dict["omega_m"]
        chain["a_s"] = wmap9Dict["a_s"]
        outfname = infname[:-4] + "_%s_blinded_%s.ini" % (ptype, pname)
    else:
        outfname = infname[:-4] + "_%s_%s.ini" % (ptype, pname)

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
    )
    out = c.analysis.get_max_posteriors()
    if do_write:
        npar_write = 0
        lines = []
        start = False
        with open(infname, "r") as infile:
            for _ in range(1000):
                ll = infile.readline()
                if ll == "## END_OF_VALUES_INI\n":
                    break
                if start:
                    lines.append(record_line(ll, out, names))
                if ll == "## START_OF_VALUES_INI\n":
                    start = True
        assert npar_write <= len(
            names
        ), "Not all the estimated parameters are writed into string, \
            something goes wrong?"
        with open(outfname, "wt") as outfile:
            outfile.write("".join(lines))
    return out

def resample_chain(samples, num_samples=None):
    """
    Resample from the given samples based on their associated weights.
    Args:
        samples (ndarray):      ndarray of samples
        num_samples (int):      number of samples to draw; defaults to the
                                number of original samples
    Returns:
        resampled ndarray of samples
    """
    if num_samples is None:
        num_samples = len(samples)
    weights = samples["weight"]
    prob = weights/np.sum(weights)
    indices = np.random.choice(len(samples), size=num_samples, p=prob)
    return samples[indices]

def record_line(ll, ss, names):
    tmp = ll.split("## ")[-1]
    nn = tmp.split(" = ")[0]
    if nn in names:
        # write the estimated parameter to the string
        if nn == "a_s":
            # the a_s in chain is scaled to by 1e9 !
            # here, we rescaled it back
            tmp = "%s = %.6E\n" % (nn, ss[nn] / 1e9)
        elif nn == "ombh2":
            tmp = "%s = %.6E\n" % (nn, ss[nn] / 1e3)
        else:
            tmp = "%s = %.6E\n" % (nn, ss[nn])
    else:
        tmp = tmp.split("##")[0]
    return tmp

def sample_datv_from_chain(infname, out_dir, nsample=100):
    """Estimate the parameters and write it to file

    Args:
        infname (str):      input file name for chain
        out_dir (str):      output dictionary name
        nsample (int):      number of generated samples
    """
    chain = read_cosmosis_chain(infname, flip_dz=False, as_correction=True)
    names = list(chain.dtype.names)
    # remove those derived parameters
    for nn in ["prior", "like", "post", "weight"]:
        names.remove(nn)
    chain = resample_chain(samples=chain, num_samples=nsample)
    for i in range(nsample):
        ss = chain[i]
        lines = []
        start = False
        with open(infname, "r") as infile:
            for _ in range(1000):
                line = infile.readline()
                if line == "## END_OF_VALUES_INI\n":
                    break
                if start:
                    lines.append(record_line(line, ss, names))
                if line == "## START_OF_VALUES_INI\n":
                    start = True
        # write results
        outfname = os.path.join(out_dir, "%02d.ini" % i)
        with open(outfname, "wt") as outfile:
            outfile.write("".join(lines))
    return

