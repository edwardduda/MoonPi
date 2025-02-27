# setup.py
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='DQNLoggerCpp',
    ext_modules=[cpp_extension.CppExtension('DQNLoggerCpp', ['DQNLoggerCpp.cpp'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)