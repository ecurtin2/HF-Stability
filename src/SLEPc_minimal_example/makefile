#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#
#  SLEPc is free software: you can redistribute it and/or modify it under  the
#  terms of version 3 of the GNU Lesser General Public License as published by
#  the Free Software Foundation.
#
#  SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
#  WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
#  FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
#  more details.
#
#  You  should have received a copy of the GNU Lesser General  Public  License
#  along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#

CPPFLAGS   = -std=c++11 -larmadillo -O3
LOCDIR     = home/evan/test/
MANSEC     = EPS
DATAPATH = ${SLEPC_DIR}/share/slepc/datafiles/matrices

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

ex3: ex3.o 
	-${CLINKER} -o ex3 ex3.o -O3 -larmadillo ${SLEPC_EPS_LIB}
	${RM} ex3.o
