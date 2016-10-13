import os
import shutil
from pyfiles.lib import CppPyxWrap
from distutils.core import setup 
from Cython.Build import cythonize 
from distutils.extension import Extension

# Don't touch this if you're not 100% sure
# seriously I know you
remove_assertions_at_compile = True
check_unused_functions = False
check_unused_variables = False
debug = False
optimize = False

WrapOut = 'HFS_CythonGenerated.pyx'

blist = ['davidson_algorithm', 'davidson_wrapper', 'mv_is_working']        
Wrap = CppPyxWrap.Wrapper(pyx_lines = 'pyfiles/HFS_pyfuncs.pyx' 
                         ,pyx_header='pyfiles/HFStability_h.pyx'
                         ,cpp_header='cppfiles/HFSnamespace.h'
                         ,func_blacklist=blist)
Wrap.combine_sections()
f = open(WrapOut, 'w')
for line in Wrap.output:
    f.write(line)
f.close()

sourcefiles  = [WrapOut, 'cppfiles/stability.cpp', 'cppfiles/HFSnamespace.cpp']
if optimize:
    compile_opts = ['-O3', '-ffast-math', '-std=c++11']
else:
    compile_opts = ['-O0',  '-std=c++11']
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
  ext_modules=cythonize(ext, gdb_debug=debug)
) 

filename, file_extension = os.path.splitext(WrapOut)
os.system('mv ' + filename + '.pyx' + ' pyfiles')
os.system('mv ' + filename + '.so'  + ' pyfiles/lib')
os.system('mv ' + filename + '.cpp' + ' cppfiles/lib')

