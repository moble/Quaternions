***NOTE:***  This repository is outdated and no longer actively developed.  I maintain a newer package [here](https://github.com/moble/quaternion), which is written in C, but also uses the [numpy](http://www.numpy.org/) C-API for much better integration with python.



Quaternions
===========

Quaternion library for C++, with python bindings via SWIG.

This code provides a simple class for calculating easily with
quaternions.  Operator overloads are provided for all the basic
operations such as addition, multiplication, etc.  The standard
functions like `exp` and `log` (which are what make quaternions so
uniquely powerful) are also available.

Additionally, support is included for time series of quaternions, as
well as operations such as differentiation, interpolation (linear and
quadratic).  Several algorithms also provide capabilities for finding
the minimal-rotation frame that tracks a certain vector, or finds the
frame that has a particular angular velocity.


Compiling with other code
=========================

This main library is fairly simple, and should be trivial to compile;
only the header `Quaternions.hpp` needs to be included, and only the
file `Quaternions.cpp` needs to be compiled.

The second pair of files, `IntegrateAngularVelocity.{c,h}pp` contain
just two functions, but depend on the GNU Scientific Library (GSL) for
interpolation and ODE integration.  So GSL must be installed
separately, compiled as a shared library, and the `-I` and `-L` flags
variables set appropriately on whatever compilation is done.

An example of how to use this code as a library in other code (and a
working Makefile) is given in the `tests` directory.

For python, compilation is done automatically, and assumes the
presence of GSL.  However, if desired, this dependency can be removed
by using the flag `--no-GSL` when compiling the python module.


Installing the python module
============================

Though this code can be included as a library in other code, it can
also be used on its own as a python module.  Just run

    python setup.py install --user

The `--user` flag installs the module to the user's home directory,
which means that no root permissions are necessary.

As mentioned above, GSL is assumed to be installed as a shared
library, and the `IncDirs` and `LibDirs` variables set appropriately
in `setup.py`.  Sensible defaults are given, so this may only need to
be done if the compilation fails.  If GSL is not installed, you can
still build most of this module by using

    python setup.py install --user --no-GSL



If the build succeeds, just open an python session and type

    import Quaternions

In ipython, you can then see your options using tab completion by typing

    Quaternions.

and then hitting tab.  Help is available on many functions in ipython
by typing a question mark after the function name.  For example:

    Quaternions.Squad?

In plain python, the same thing can be achieved by entering
`help(Quaternions.Squad)`.  But if you're using plain python
interactively, you should really give ipython a try.
