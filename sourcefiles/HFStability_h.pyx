# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False

from libcpp cimport bool
from libc.math cimport sqrt
import itertools
import math
from scipy import special as sp
import numpy as np
cimport numpy as np
cimport cython
import matplotlib.pyplot as plt
import seaborn as sns

# This group imports from ./lib
include "lib/cyarma.pyx"    
from lib import general_methods as gm
