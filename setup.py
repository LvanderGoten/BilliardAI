from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("billiard_optimized",
                         ["billiard_optimized.pyx"],
                         libraries=["m"],
                         extra_compile_args=["-ffast-math"])]

setup(
    name="billiard_optimized",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules)
