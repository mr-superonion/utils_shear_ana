import os
from setuptools import setup, find_packages

# version of the package
__version__ = ""
fname = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "utils_shear_ana", "__version__.py"
)
with open(fname, "r") as ff:
    exec(ff.read())


scripts = [
        "bin/make_mock_datav.py",
        "bin/make_cosmosis_ini.py",
        "bin/submit_shear_job.py",
        "bin/shear_config",
        ]

setup(
    name="utils_shear_ana",
    version=__version__,
    description="shear analysis utilities",
    author="Xiangchong Li",
    author_email="mr.superonion@hotmail.com",
    python_requires=">=3.9",
    install_requires=[
        "astropy",
        "matplotlib",
        "getdist",
        "chainconsumer",
        "healpy",
        "tensiometer",
        "treecorr",
    ],
    packages=find_packages(),
    scripts=scripts,
    include_package_data=True,
    zip_safe=False,
    url="https://github.com/mr-superonion/utils_shear_ana/",
)
