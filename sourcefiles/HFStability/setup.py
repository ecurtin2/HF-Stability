import os
import shutil
from distutils.core import setup 
from Cython.Build import cythonize 
from distutils.extension import Extension

# Don't touch this if you're not 100% sure
# seriously I know you
remove_assertions_at_compile = True
check_unused_functions = False
check_unused_variables = False

sourcefiles  = ['HFS.pyx', 'stability.cpp', 'HFSnamespace.cpp']
compile_opts = ['-O3', '-ffast-math', '-std=c++11']
my_libraries = ['armadillo']

if not check_unused_variables:
    compile_opts.append('-Wno-unused-variable')
if not check_unused_functions:
    compile_opts.append('-Wno-unused-function')

# This section removes wstrict prototypes
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")


if remove_assertions_at_compile:
    compile_opts.append('-DPYREX_WITHOUT_ASSERTIONS')

ext=[Extension('*', sourcefiles, extra_compile_args=compile_opts, language='c++', libraries=my_libraries)] 
 
setup( 
  ext_modules=cythonize(ext, gdb_debug=True)
) 
