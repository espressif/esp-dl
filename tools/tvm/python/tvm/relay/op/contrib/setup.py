# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='esp',
    ext_modules=cythonize("esp.py"),
    zip_safe=False,
)

