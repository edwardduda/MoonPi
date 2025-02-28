from setuptools import setup, Extension
import pybind11
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Extension(
        "MarketEnvCpp",  # Name of the module
        ["MarketEnvCpp.cpp"],  # C++ source files
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-std=c++14", "-O3"]
    ),
]

setup(
    name="MarketEnvCpp",
    ext_modules=ext_modules,
)