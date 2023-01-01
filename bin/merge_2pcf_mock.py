#!/usr/bin/env python
# Copyright 20220312 Xiangchong Li.
# This task measure the two point correlation function for real space
# cosmic shear from mocks
import os
import glob
import fitsio
import argparse
import schwimmbad
import numpy as np


class Worker(object):
    def __init__(self, datname, fieldname, do_finer):
        self.nz = 4
        wrkDir = os.environ["homeWrk"]
        self.fieldname = fieldname
        self.do_finer = do_finer
        if not do_finer:
            self.cor_ver = "cor_fields"
        else:
            self.cor_ver = "cor_finer"
        self.blind_ver = datname
        self.corDir = os.path.join(
            wrkDir, "cosmicShear/mocksim/%s_%s/" % (self.cor_ver, self.blind_ver)
        )
        return

    def run(self, ref):
        isim = ref // 13
        irot = ref % 13
        # correlation version
        flist = glob.glob(
            os.path.join(
                self.corDir,
                "r%03d_rotmat%d_%s_cor*.fits" % (isim, irot, self.fieldname),
            )
        )
        if len(flist) != self.nz * (self.nz + 1) / 2:
            print(
                "Do not have all the simulations for ",
                "isim: %d, irot: %d, field: %s" % (isim, irot, self.fieldname),
                "under %s" %(self.corDir),
            )
            return

        dd_all = []
        # multiplicative bias table
        for i in range(self.nz):
            for j in range(i, self.nz):
                fname = os.path.join(
                    self.corDir,
                    "r%03d_rotmat%d_%s_cor%d%d.fits"
                    % (isim, irot, self.fieldname, i + 1, j + 1),
                )
                data = fitsio.read(fname)
                dd = np.hstack([data["xip"], data["xim"]])
                # if not np.all((np.abs(dd) > 1e-20) & (np.abs(dd) < 1e-3)):
                #     print(
                #         "Find a problematic simulation: isim: %d, irot: %d, field: %s"
                #         % (isim, irot, self.fieldname)
                #     )
                dd_all.append(dd)
                del data
        dd_all = np.stack(dd_all)
        return dd_all

    def __call__(self, ref):
        return self.run(ref)


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
    outputs = []
    for r in pool.map(worker, refs):
        outputs.append(r)
    pool.close()
    outputs = np.stack(outputs)
    fname = os.path.join(
        os.environ["homeWrk"],
        "cosmicShear/mocksim/%s_%s_%s.fits"
        % (worker.cor_ver, args.field, worker.blind_ver),
    )
    fitsio.write(fname, outputs)
