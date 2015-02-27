from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

multiscale_sourcefiles = ["pymultiscale/dwt.pyx", "pymultiscale/dwt1.cc",
                       "pymultiscale/dwt2.cc", "pymultiscale/dwt3.cc"]
multiscale_extension = Extension("pymultiscale.dwt",
                        multiscale_sourcefiles,
                        language="c++",
                        include_dirs=[np.get_include()],
                        libraries=[],
                        extra_compile_args=['-O3', '-Wno-unused-function'],
                        extra_link_args=['-L/usr/local/lib']
                        )

extensions = [multiscale_extension]

# Set up the module
setup(name='pymultiscale',
      version='0.1',
      description='Multiscale transforms for Python in 1D, 2D, and 3D',
      url='https://github.com/broxtronix/pymultiscale',
      author='Michael Broxton',
      author_email='broxton@gmail.com',
      license='MIT',
      packages=['pymultiscale'],
      zip_safe=False,
      ext_modules = cythonize(extensions))

