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


def make_config_ini(
    outfname,
    chain_name="fid",
    blind_name="cat0",
    sampler="multinest",
    modules=None,
):
    """Makes configuration ini file

    Args:
        outfname (str):     output ini file name
        chain_name (str):   chain name
        blind_name (str):   blinding name
        sampler (str):      sampler name
        modules (str):      names of modules
    """
    if modules is None:
        modules = "free_params_sig8    consistency\
            bbn_consistency     camb_sig8\
            load_nz             nzbias\
            ia                  ia_z\
            pk2cl               add_ia\
            shear_m_bias        cl2xi\
            add_sys             2pt_like"

    content = "[DEFAULT]\n\
confDir=$cosmosis_utils/config/s19a/\n\
runname=%s\n\
blindname=%s\n\
\n\
[runtime]\n\
sampler = %s\n\
\n\
[pipeline]\n\
fast_slow = T\n\
values = %%(confDir)s/pars/%%(runname)s_values.ini\n\
priors = %%(confDir)s/pars/%%(runname)s_priors.ini\n\
\n\
modules = %s\n\
\n\
extra_output = cosmological_parameters/S_8\n\
quiet=T\n\
debug=F\n\
\n\
[output]\n\
filename = outputs/out_%%(runname)s.txt\n\
format=text\n\
privacy = F\n\
\n\
%%include $cosmosis_utils/config/s19a/models/mc.ini\n\
%%include $cosmosis_utils/config/s19a/models/cosmo.ini\n\
%%include $cosmosis_utils/config/s19a/models/astro.ini\n\
%%include $cosmosis_utils/config/s19a/models/sys.ini\n\
%%include $cosmosis_utils/config/s19a/scales/scale_cut_fid.ini\n\
"
    with open(outfname, "wt") as outfile:
        outfile.write(content%(chain_name, blind_name, sampler, modules))
    return
