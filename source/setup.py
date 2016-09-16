from distutils.core import setup 
from Cython.Build import cythonize 
from distutils.extension import Extension
import cppheader_to_pyxheader

# Don't touch this if you're not 100% sure
# seriously I know you
remove_assertions_at_compile = False

#rewrites the c class wrapping .pyx part at before compiling
cppheader_to_pyxheader.cpp_head_to_pyxhead('stability', breaker='davidson')
        

sourcefiles  = ['HFStability.pyx', 'stability.cpp']
compile_opts = ['-O3', '-ffast-math', '-std=c++11']
my_libraries = ['armadillo']

if remove_assertions_at_compile:
    compile_opts.append('-DPYREX_WITHOUT_ASSERTIONS')

ext=[Extension('*', sourcefiles, extra_compile_args=compile_opts, language='c++', libraries=my_libraries)] 
 
setup( 
  ext_modules=cythonize(ext) 
) 
