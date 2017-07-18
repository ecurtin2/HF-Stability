from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import shutil
import glob
import sys
import os

sys.argv.append('build_ext')
sys.argv.append('--inplace')

if 'clean' in sys.argv:
    for f in glob.glob('*.so'):
        os.remove(f)
    try:
        os.remove('CppRowGen.cpp')
    except:
        pass
    try:
        shutil.rmtree('build')
    except:
        pass
    sys.exit()


extensions = [Extension('CppRowGen'
                ,['CppRowGen.pyx']
                ,extra_compile_args=['-std=c++11', '-O3']
                ,language='c++'
                )
             ]
setup(ext_modules=cythonize(extensions)
)

shutil.move('CppRowGen.cpp', 'build')

files = glob.glob('*.so')
if len(files) == 0:
    raise FileNotFoundError('No shared object was built during setup!')
elif len(files) > 1:
    raise FileExistsError('Too many shared objects in directory (only 1 supported currently).')
else:
    shutil.move(files[0], 'CppRowGen.so')

