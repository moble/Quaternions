#! /usr/bin/env python

"""
Installation file for python code associated with the paper "Angular
velocity of gravitational radiation from precessing binaries and the
corotating frame".

To build this code and run it in place, run
    python setup.py build_ext --inplace
then open python and type 'import Quaternions' in
the current directory.

To install in the user's directory, run
    python setup.py install --user
Now, 'import Quaternions' may be run from a python
instance started in any directory on the system.
"""

## Check for `--no-GSL` option
from sys import argv
if '--no-GSL' in argv:
    GSL=False
    GSLDef = ''
    argv.remove('--no-GSL')
else:
    GSL=True
    GSLDef = '-DUSE_GSL'

## If PRD won't let me keep a subdirectory, make one
from os.path import exists
from os import makedirs
if not exists('Quaternions') :
    makedirs('Quaternions')
# from shutil import copyfile
# if not exists('Quaternions/plot.py') :
#     copyfile('plot.py', 'Quaternions/plot.py')

## distutils doesn't build swig modules in the correct order by
## default -- the python module is installed first.  This will pop
## 'build_ext' to the beginning of the command list.
from distutils.command.build import build
build.sub_commands = sorted(build.sub_commands, key=lambda sub_command: int(sub_command[0]!='build_ext'))

## We also need to copy the SWIG-generated python script Quaternions.py
## to Quaternions/__init__.py so that it gets installed correctly.
from distutils.command.build_ext import build_ext as _build_ext
from distutils.file_util import copy_file
class build_ext(_build_ext):
    """Specialized Python source builder for moving SWIG module."""
    def run(self):
        _build_ext.run(self)
        copy_file('Quaternions.py', 'Quaternions/__init__.py')

## Now import the basics
from distutils.core import setup, Extension
from subprocess import check_output, CalledProcessError
from os import devnull, environ

# If /opt/local directories exist, use them
from os.path import isdir
if isdir('/opt/local/include'):
    IncDirs = ['/opt/local/include']
else:
    IncDirs = []
if isdir('/opt/local/lib'):
    LibDirs = ['/opt/local/lib']
else:
    LibDirs = []

# Add directories for numpy inclusion
from numpy import get_include
IncDirs += [get_include()]

# Add directories for GSL, if needed
if GSL :
    SourceFiles = ['Quaternions.cpp',
                   'Utilities.cpp',
                   'IntegrateAngularVelocity.cpp',
                   'Quaternions.i']
    Dependencies = ['Quaternions.hpp',
                    'Utilities.hpp',
                    'IntegrateAngularVelocity.hpp',
                    'Quaternions_typemap.i']
    Libraries = ['gsl', 'gslcblas']
    ## See if GSL_HOME is set; if so, use it
    if "GSL_HOME" in environ :
        IncDirs = [environ["GSL_HOME"]+'/include'] + IncDirs
        LibDirs = [environ["GSL_HOME"]+'/lib'] + IncDirs
else :
    SourceFiles = ['Quaternions.cpp',
                   'Quaternions.i']
    Dependencies = ['Quaternions.hpp',
                    'Quaternions_typemap.i']
    Libraries = []

## Remove a compiler flag that doesn't belong there for C++
import distutils.sysconfig as ds
cfs=ds.get_config_vars()
for key, value in cfs.items() :
    if(type(cfs[key])==str) :
        cfs[key] = value.replace('-Wstrict-prototypes', '')

## Read in the license
try :
    with open('LICENSE', 'r') as myfile :
        License=myfile.read()
except IOError :
    License = 'See LICENSE file in the source code for details.'

swig_opts=['-globals', 'constants', '-c++', '-builtin', GSLDef]
try:
    import sys
    python_major = sys.version_info.major
    if(python_major==3) :
        swig_opts += ['-py3']
except AttributeError:
    print("Your version of python is SO old.  'How old is it?'  So old I can't even tell how old it is.")
    print("No, seriously.  You should think about upgrading your python because I don't support this version.")
    print("You can try to make this run by removing the assertion error you're about to get, but don't")
    print("come crying to me when print statements fail or when division gives the wrong answer.")
    raise AssertionError("Wake up grandpa!  You were dreaming of ancient pythons again.")

## This does the actual work
setup(name="Quaternions",
      # version=PackageVersion,
      description='Quaternion library for C++, with python bindings via SWIG.',
      #long_description=""" """,
      author='Michael Boyle',
      author_email='boyle@astro.cornell.edu',
      url='https://github.com/MOBle/Quaternions',
      license=License,
      packages = ['Quaternions'],
      ext_modules = [
        Extension('_Quaternions',
                  sources=SourceFiles,
                  depends=Dependencies,
                  include_dirs=IncDirs,
                  library_dirs=LibDirs,
                  libraries=Libraries,
                  #define_macros = [('CodeRevision', CodeRevision)],
                  language='c++',
                  swig_opts=swig_opts,
                  extra_link_args=['-fPIC'],
                  extra_compile_args=['-Wno-deprecated', # Numpy compilations always seem to involve deprecated things
                                      '-ffast-math', # NB: fast-math makes it impossible to detect NANs
                                      GSLDef]
                  )
        ],
      # classifiers = ,
      # distclass = ,
      # script_name = ,
      # script_args = ,
      # options = ,
      # license = ,
      # keywords = ,
      # platforms = ,
      # cmdclass = ,
      cmdclass={'build_ext': build_ext},
      # data_files = ,
      # package_dir =
      )
