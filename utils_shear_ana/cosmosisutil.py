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
import logging

modules_default = "free_params_sig8    consistency  \n\
    bbn_consistency     camb_sig8 \n\
    load_nz             nzbias \n\
    ia                  ia_z \n\
    pk2cl               add_ia \n\
    shear_m_bias        cl2xi \n\
    add_sys             2pt_like"


def make_config_ini(
    runname="fid",
    datname="cat0",
    psfname="cat0",
    fieldname="all",
    sampler="multinest",
    sid=1,
    modules=None,
    valuename="fid",
    priorname="fid",
    **kargs,
):
    """Makes configuration ini file to analysis data

    Args:
        runname (str):   chain name
        datname (str):      data version name [cat0, cat1 or simulations]
        sampler (str):      sampler name
        modules (str):      names of modules
        valuename (str):    value file name
        priorname (str):    prior file name
    """
    if modules is None:
        modules = modules_default

    content = "[DEFAULT]\n\
confDir=$cosmosis_utils/config/s19a/\n\
runname=%s\n\
datname=%s\n\
psfname=%s\n\
fieldname=%s\n\
pname=\n\
\n\
[runtime]\n\
sampler = %s\n\
\n\
[pipeline]\n\
fast_slow = T\n\
values = %%(confDir)s/pars/%s_values.ini\n\
priors = %%(confDir)s/pars/%s_priors.ini\n\
\n\
modules = %s\n\
\n\
extra_output = cosmological_parameters/S_8 data_vector/2pt_chi2 cosmological_parameters/sigma_8\n\
quiet=T\n\
debug=F\n\
\n\
[output]\n\
filename = outputs/%s_%%(runname)s_%%(datname)s.txt\n\
format=text\n\
privacy = F\n\
\n\
%%include $cosmosis_utils/config/s19a/models/mc_%d.ini\n\
%%include $cosmosis_utils/config/s19a/models/cosmo.ini\n\
%%include $cosmosis_utils/config/s19a/models/astro.ini\n\
%%include $cosmosis_utils/config/s19a/models/sys.ini\n\
%%include $cosmosis_utils/config/s19a/models/likelihood.ini\n\
"
    assert os.path.isdir("configs")
    outfname = "configs/%s_%s_%s.ini" % (sampler, runname, datname)
    if not os.path.isfile(outfname):
        if sampler[-1] in ["2", "3", "4"]:
            sampler0 = sampler[:-1]
        else:
            sampler0 = sampler
        with open(outfname, "wt") as outfile:
            outfile.write(
                content
                % (
                    runname,
                    datname,
                    psfname,
                    fieldname,
                    sampler0,
                    valuename,
                    priorname,
                    modules,
                    sampler,
                    sid,
                )
            )
    else:
        logging.warn("Already has output ini file: %s" % outfname)
    return


def make_config_sim_ini(
    runname="fid",
    datname="cowls85",
    psfname="cat0",
    sampler="multinest",
    sid=1,
    modules=None,
    valuename="fid",
    priorname="fid",
    **kargs,
):
    """Makes configuration ini file to analysis simulation

    Args:
        runname (str):      chain name
        datname (str):      data version name [cat0, cat1 or simulations]
        sampler (str):      sampler name
        modules (str):      names of modules
        valuename (str):    value file name
        priorname (str):    prior file name
    """
    if modules is None:
        modules = modules_default

    content = "[DEFAULT]\n\
confDir=$cosmosis_utils/config/s19a/\n\
runname=%s\n\
datname=%s\n\
psfname=%s\n\
fieldname=all\n\
pname=\n\
\n\
[runtime]\n\
sampler = %s\n\
\n\
[pipeline]\n\
fast_slow = T\n\
values = %%(confDir)s/pars/%s_values.ini\n\
priors = %%(confDir)s/pars/%s_priors.ini\n\
\n\
modules = %s\n\
\n\
extra_output = cosmological_parameters/S_8 data_vector/2pt_chi2 cosmological_parameters/sigma_8\n\
quiet=T\n\
debug=F\n\
\n\
[output]\n\
filename = outputs/%s_%%(runname)s_%%(datname)s.txt\n\
format=text\n\
privacy = F\n\
\n\
%%include $cosmosis_utils/config/s19a/models/mc_%d.ini\n\
%%include $cosmosis_utils/config/s19a/models/cosmo.ini\n\
%%include $cosmosis_utils/config/s19a/models/astro.ini\n\
%%include $cosmosis_utils/config/s19a/models/sys.ini\n\
%%include $cosmosis_utils/config/s19a/models/likelihood.ini\n\
"
    assert os.path.isdir("configs")
    outfname = "configs/%s_%s_%s.ini" % (sampler, runname, datname)
    if not os.path.isfile(outfname):
        if sampler[-1] in ["2", "3", "4"]:
            sampler0 = sampler[:-1]
        else:
            sampler0 = sampler
        with open(outfname, "wt") as outfile:
            outfile.write(
                content
                % (
                    runname,
                    datname,
                    psfname,
                    sampler0,
                    valuename,
                    priorname,
                    modules.replace("2pt_like", "2pt_like_sim"),
                    sampler,
                    sid,
                )
            )
    else:
        logging.warn("Already has output ini file: %s" % outfname)
    return
