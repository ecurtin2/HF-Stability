from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import shutil
import glob
import sys
import os


def main():
    # hack this so I don't have to type stuff when running setup.py
    sys.argv += ['build_ext', '--inplace']
    
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
    for e in extensions:
        e.cython_directives = {
            'boundscheck': False,
            'wraparound' : False,
            'cdivision'  : True,
            'initializedcheck': False,
        }
    setup(ext_modules=cythonize(extensions, annotate=True)
    )

    
    shutil.move('CppRowGen.cpp', 'build')
    
    files = glob.glob('*.so')
    if len(files) == 0:
        raise FileNotFoundError('No shared object was built during setup!')
    elif len(files) > 1:
        raise FileExistsError('Too many shared objects in directory (only 1 supported currently).')
    else:
        shutil.move(files[0], 'CppRowGen.so')
 
if __name__ == '__main__':
    main()
