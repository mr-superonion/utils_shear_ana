#!/usr/bin/env python
# Copyright 20220312 Xiangchong Li.
# This task measure the two point correlation function for real space
# cosmic shear from mocks
import gc
import os
import glob
import fitsio
import argparse
import schwimmbad
import numpy as np
from utils_shear_ana import catutil
from utils_shear_ana import mea2pcf
import astropy.table as astTable

# correction for shell thickness
corrs = np.array([1.17133725, 1.08968149, 1.06929737, 1.05591374])


class Worker(object):
    def __init__(self, datname, fieldname, do_finer):
        self.nz = 4
        self.blind_ver = datname
        self.fieldname = fieldname
        wrkDir = os.environ["homeWrk"]
        self.do_finer = do_finer
        if not self.do_finer:
            self.cor_ver = "cor_fields"
        else:
            self.cor_ver = "cor_finer"
        self.oDir = os.path.join(
            wrkDir, "cosmicShear/mocksim/%s_%s/" % (self.cor_ver, self.blind_ver)
        )
        if not os.path.isdir(self.oDir):
            os.makedirs(self.oDir, exist_ok=True)
        self.mockDir = os.path.join(wrkDir, "S19ACatalogs/catalog_mock/shape_v2/")
        self.msklist = []
        for i in range(self.nz):
            _ = os.path.join(
                os.environ["homeWrk"], "cosmicShear/catalog/field_zbin%d.fits" % (i + 1)
            )
            if fieldname != "all":
                self.msklist.append(fitsio.read(_)["field"] == fieldname)
            else:
                self.msklist.append(fitsio.read(_)["field"] != "nan")
                # self.msklist.append(~np.load("./msk_%d.npy" %i))
        # for i in range(self.nz):
        #     print(np.sum(~self.msklist[i]))
        self.mrTab = astTable.Table.read(
            os.path.join(os.environ["HOME"], "hsc_blinds/shear_biases_fiducial.csv")
        )
        return

    def read_data(self, fname, iz):
        dd = fitsio.read(fname)
        dd = dd[self.msklist[iz]]
        dd = catutil.make_mock_catalog(
            dd,
            mbias=self.mrTab["m_shear_%s" % self.blind_ver][iz],
            msel=self.mrTab["m_sel"][iz],
            corr=corrs[iz],
        )
        msk = (dd["noise1_int"] ** 2.0 + dd["noise2_int"] ** 2.0) < 10.0
        msk = msk & ((dd["noise1_mea"] ** 2.0 + dd["noise2_mea"] ** 2.0) < 10.0)
        dd = dd[msk]
        del msk
        cat = mea2pcf.convert_mock2treecat(
            dd, self.mrTab["m_shear_%s" % self.blind_ver][iz], self.mrTab["m_sel"][iz]
        )
        return cat

    def run(self, ref):
        isim = ref // 13
        irot = ref % 13
        # correlation version
        if not self.do_finer:
            cor = mea2pcf.corDF
        else:
            cor = mea2pcf.corB360
        flist = glob.glob(
            os.path.join(
                self.oDir, "r%03d_rotmat%d_%s_cor*.fits" % (isim, irot, self.fieldname)
            )
        )
        if len(flist) == self.nz * (self.nz + 1) / 2:
            print(
                "Already have all the fiels for isim: %d, irot: %d, field: %s \n\
                at %s"
                % (isim, irot, self.fieldname, self.oDir)
            )
            return

        for i in range(self.nz):
            znmi = os.path.join(
                self.mockDir,
                "fiducial_zbins/cat_r%03d_rotmat%d_zbin%d.fits" % (isim, irot, i + 1),
            )
            catI = self.read_data(znmi, i)
            for j in range(i, self.nz):
                znmj = os.path.join(
                    self.mockDir,
                    "fiducial_zbins/cat_r%03d_rotmat%d_zbin%d.fits"
                    % (isim, irot, j + 1),
                )
                catJ = self.read_data(znmj, j)
                cor.clear()
                cor.process(catI, catJ)
                _ofname = os.path.join(
                    self.oDir,
                    "r%03d_rotmat%d_%s_cor%d%d.fits"
                    % (isim, irot, self.fieldname, i + 1, +j + 1),
                )
                cor.write(_ofname)
                del catJ
                gc.collect()
            del catI
            gc.collect()
        return

    def __call__(self, ref):
        self.run(ref)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="correlation fucntion")
    parser.add_argument(
        "--minId", required=True, type=int, help="minimum id number, e.g. 0"
    )
    parser.add_argument(
        "--maxId", required=True, type=int, help="maximum id number, e.g. 1404"
    )
    parser.add_argument(
        "--field",
        required=True,
        type=str,
        help="field name, all, XMM, VVDS or GAMA09H, HECTOMAP etc.",
    )
    parser.add_argument(
        "--datname", default="cat0", type=str, help="data name. cat0, cat1 or cat2"
    )
    parser.add_argument(
        "--finer", default=False, type=bool, help="whether do finer for B-mode test"
    )
    # mpi
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ncores",
        dest="n_cores",
        default=1,
        type=int,
        help="Number of processes (uses multiprocessing).",
    )
    group.add_argument(
        "--mpi", dest="mpi", default=False, action="store_true", help="Run with MPI."
    )
    args = parser.parse_args()

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    worker = Worker(args.datname, args.field, args.finer)
    refs = list(range(args.minId, args.maxId))
    for r in pool.map(worker, refs):
        pass
    pool.close()
