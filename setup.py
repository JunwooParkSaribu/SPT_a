from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Hello world app',
    ext_modules=cythonize("test_cy.pyx", annotate=True),
)

#  python3 setup.py build_ext --inplace