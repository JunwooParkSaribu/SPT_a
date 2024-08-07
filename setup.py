from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Hello world app',
    ext_modules=cythonize("./module/image_pad.pyx", language_level = "3", annotate=True),
)

#  python3 setup.py build_ext --inplace