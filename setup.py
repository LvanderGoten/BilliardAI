from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

ext_modules = [Extension("billiard",
                         ["billiard.pyx"],
                         libraries=["m"],
                         extra_compile_args=["-ffast-math"])]

setup(
    name="billiard",
    version="0.0.1",
    install_requires=["arcade>=2.0.0a3", "numpy>=1.15.3", "gym>=0.10.9"],
    ext_modules=cythonize(ext_modules)
)

