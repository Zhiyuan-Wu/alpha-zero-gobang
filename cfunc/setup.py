from setuptools import setup, Extension
import numpy

# Define the extension module name and sources
module_name = 'cfunc'
module_sources = ['cfunc.c']

# Define the extension module object with numpy include directories
module = Extension(module_name, sources=module_sources, include_dirs=[numpy.get_include()])

# Define the setup parameters
setup(name=module_name,
      version='1.0',
      description='A module that provides a C function for numpy array sum.',
      ext_modules=[module])