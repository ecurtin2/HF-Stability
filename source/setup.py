from distutils.core import setup 
from Cython.Build import cythonize 
from distutils.extension import Extension

sourcefiles  = ['*.pyx', 'stability.cpp']
compile_opts = ['-O3', '-ffast-math', '-std=c++11']
my_libraries = ['armadillo']

ext=[Extension('*', sourcefiles, extra_compile_args=compile_opts, language='c++', libraries=my_libraries)] 
 
setup( 
  ext_modules=cythonize(ext) 
) 

#import os
#os.system('mv *.so ~/usr/bin/python')
