## Make ini files

### for real data
```shell
make_cosmosis_ini.py -d cat0 --num -1 --inds 0 3 {-s minuit or multinest}
```

### for mock simulations
```shell
make_cosmosis_ini.py -d camb --num -1 --inds 0 3 {-s minuit or multinest}
```

```shell
make_cosmosis_ini.py -d camb --num 10 --inds 0 3 {-s minuit or multinest}
```

## submit jobs

### minuit
submit_shear_job.py configs/minuit_* --queue tiny

### multinest
submit_shear_job.py configs/multinest_* --queue mini
