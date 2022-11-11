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
import gc
import fitsio
import numpy as np
from . import catutil


def bin_catalog_inz(catFname):
    """Loads catalog [mock or real] from all fields, combine them, and divide
    them into 4 redshift bins.

    Parameters:
        catFname (str): catalog name divided into fields [replace field name with %]
    Returns:
        dataG (list):   list of catalogs divided into 4 redshift bins
    """
    nzs = 4
    assert "%" in catFname, "% not in catFname"
    wrkDir = os.environ["homeWrk"]
    # catFname=   os.path.join(wrkDir,catFname)
    # 4 bins
    dataG = []
    for _ in range(nzs):
        dataG.append([])
    for fieldname in catutil.field_names:
        # mock catalog (will copy v2 to GW and idark)
        _tmpnm = catFname % fieldname
        assert os.path.isfile(_tmpnm), _tmpnm
        data = fitsio.read(_tmpnm)
        # selection using measured photo-z (not the true source photo-z)
        pzsname = os.path.join(
            wrkDir,
            "S19ACatalogs/photoz_2pt/fiducial_dnnzbin_w95c027/source_sel_%s.fits"
            % fieldname,
        )
        binarray = fitsio.read(pzsname)
        assert np.all(binarray["object_id"] == data["object_id"])
        for iz in range(nzs):
            _msk = binarray["dnnz_bin"] == (iz + 1)
            # add field to list
            dataG[iz].append(data[_msk])
            del _msk
        del binarray, data
        gc.collect()
    dataG = [np.hstack(dataG[i]) for i in range(nzs)]
    return dataG


def bin_catalog_inz2(catFname, outFname, zbin):
    """Loads catalog [mock or real] from all fields, combine them, and divide
    them into the zbin's redshift bin

    Args:
        catFname (str):     catalog name divided into fields [replace field name with %]
        outFname (str):     file name for outcome
        zbin (int):         bin number
    """
    assert zbin in [1, 2, 3, 4], "zbin should be either 1,2,3 or 4"
    assert "%" in catFname
    wrkDir = os.environ["homeWrk"]
    # catFname=   os.path.join(wrkDir,catFname)
    # outFname=   os.path.join(wrkDir,outFname)
    # 4 bins
    dataG = []
    for fieldname in catutil.field_names:
        # mock catalog
        _tmpnm = catFname % fieldname
        assert os.path.isfile(_tmpnm)
        data = fitsio.read(_tmpnm)
        # selection using measured photo-z (not the true source photo-z)
        pzsname = os.path.join(
            wrkDir,
            "S19ACatalogs/photoz_2pt/fiducial_dnnzbin_w95c027/source_sel_%s.fits"
            % fieldname,
        )
        binarray = fitsio.read(pzsname)
        assert np.all(binarray["object_id"] == data["object_id"])
        _msk = binarray["dnnz_bin"] == zbin
        del binarray
        # add field to list
        dataG.append(data[_msk])
        del _msk, data
        gc.collect()
    dataO = np.hstack(dataG)
    fitsio.write(outFname, dataO)
    return


def bin_catalog_inz3x2pt(catFname, outFname):
    """Loads catalog (mock or real) from all fields, combine them, and divide
    them into the zbin's redshift bin

    Args:
        catFname (str):     catalog name divided into fields (replace field name with %)
        outFname (str):     file name for outcome
    """
    assert "%" in catFname
    wrkDir = os.environ["homeWrk"]
    catFname = os.path.join(wrkDir, catFname)
    outFname = os.path.join(wrkDir, outFname)
    # 4 bins
    dataG = []
    for fieldname in catutil.field_names:
        # mock catalog (will copy v2 to GW and idark)
        assert os.path.isfile(catFname % fieldname)
        data = fitsio.read(catFname % fieldname)

        # selection using measured photo-z (not the true source photo-z)
        pzsname = os.path.join(
            wrkDir,
            "cosmicShear/tpcf/from_sunao/cosmic-shear-meas-s19a/3x2pt/source_sel_dnnz_%s.fits"
            % fieldname,
        )
        binarray = fitsio.read(pzsname)
        assert np.all(binarray["object_id"] == data["object_id"])
        # add field to list
        dataG.append(data[binarray["z075"]])
        del data, binarray
        gc.collect()
    dataO = np.hstack(dataG)
    fitsio.write(outFname, dataO)
    return
