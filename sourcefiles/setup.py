import os
import shutil
from distutils.core import setup 
from Cython.Build import cythonize 
from distutils.extension import Extension
from lib import ClassBlender

# Don't touch this if you're not 100% sure
# seriously I know you
remove_assertions_at_compile = False
check_unused_functions = False
check_unused_variables = False

#rewrites the c class wrapping .pyx part at before compiling
a = ClassBlender.CppClassWrapper(pyx_header='HFStability_h.pyx', pyx_class='HFStability_class.pyx', 
                    cpp_header='stability.h',       cpp_class='stability.cpp')
a.combine_sections()
f = open('ClassWrap.pyx', 'w')
for line in a.output:
    f.write(line)
f.close()

sourcefiles  = ['ClassWrap.pyx', 'stability.cpp']
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
  ext_modules=cythonize(ext) 
) 

shutil.move('ClassWrap.so', 'lib/ClassWrap.so')       
shutil.move('ClassWrap.pyx', 'lib/ClassWrap.pyx')       
shutil.move('ClassWrap.cpp', 'lib/ClassWrap.cpp')       
