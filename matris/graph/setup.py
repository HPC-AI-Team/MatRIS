
from setuptools import Extension, setup

ext_modules = [Extension("matris.graph.cygraph", ["matris/graph/cygraph.pyx"])]

setup(ext_modules=ext_modules, setup_requires=["Cython"])

# python setup.py build_ext --inplace