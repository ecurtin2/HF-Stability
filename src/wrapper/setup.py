from distutils.core import setup, Extension

modulename = 'HFS'

sourcefiles = ['HFS.i'  # SWIG instruction file
              ,'../cppfiles/HFS_main.cpp'
              ,'../cppfiles/HFS_params.cpp'
              ,'../cppfiles/HFS_base_funcs.cpp'
              ,'../cppfiles/HFS_params_calc.cpp'
              ,'../cppfiles/HFS_matrix_utils.cpp'
              ,'../cppfiles/HFS_davidson.cpp'
              ,'../cppfiles/HFS_debug.cpp'
              ,'../cppfiles/HFS_fileIO.cpp']


swigops = ['-c++', '-ignoremissing','-w509']

# Don't touch this if you're not 100% sure
# seriously I know you
remove_assertions_at_compile = True
check_unused_functions = False
check_unused_variables = False
debug = False
optimize = False
my_libraries = ['armadillo', 'tcl']
compile_opts = ['-O3', '-ffast-math', '-std=c++11']


################################################################################
#                                                                              #
#                DONT CHANGE BELOW UNLESS YOU KNOW WHAT'S UP                   #
#                                                                              #        
################################################################################

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

modules=[Extension('_'+modulename, sourcefiles, extra_compile_args=compile_opts,
                    language='c++', libraries=my_libraries, swig_opts=swigops)] 
setup(name=modulename, ext_modules=modules, py_modules=[modulename])
