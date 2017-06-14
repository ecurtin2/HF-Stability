#
# 'make depend'  uses makedepend to automatically generate dependencies
#                (dependencies are added to end of Makefile)
# 'make debug2D'   Compile with debugging flags
# 'make debug3D'   Compile with debugging flags
# 'make profile2D' Compile with Profile flags
# 'make profile3D' Compile with Profile flags
# 'make release2D' Compile with Release flags
# 'make release3D' Compile with Release flags
# 'make clean'     clean bin/ and build/
#
CC = g++
SRCDIR = src
SRCEXT = cpp
TARGET = HFS

LIB = -larmadillo
LOCDIR = home/evan/git/HF-Stability/
MANSEC = EPS

CFLAGS = -Wall -std=c++11 -fPIC
DATAPATH = $(SLEPC_DIR)/share/slepc/datafiles/matrices
INC = -I$(SLEPC_DIR)/include/ -I$(SLEPC_DIR)/$(SLEPC_ARCH)/include/ -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include/ -I$(BOOST_ROOT)

SOURCES = $(wildcard $(SRCDIR)/*$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,%,$(SOURCES:.$(SRCEXT)=.o))

COMP = Work

# Work
ifeq ($(COMP),Work)
        CC=/home/evan/local/anaconda3/bin/g++
	SLEPC_DIR=/home/evan/local/slepc-3.7.3
	PETSC_DIR=/home/evan/git/petsc
	PETSC_ARCH=linux-gnu-c-debug
	SLEPC_ARCH=linux-gnu-c-debug
	BOOST_ROOT=/home/evan/local/boost_1_63_0
endif

# Gellmann
ifeq ($(COMP),Gellmann)
	CC = /share/ccs/gcc/gcc-5.4.0/bin/g++
	SLEPC_DIR=/home/ecurtin2/local/lib/slepc-3.7.3
	PETSC_DIR=/home/ecurtin2/local/lib/petsc
	PETSC_ARCH=arch-64-bit-indices
	SLEPC_ARCH=arch-64-bit-indices
	LOCDIR=home/ecurtin2/git/HF-Stability/
	BOOST_ROOT=/home/ecurtin2/local/boost_1_63_0
endif

ifeq ($(COMP),Home)
	SLEPC_DIR=/home/evan/git/slepc-3.7.3
	PETSC_DIR=/home/evan/git/petsc
	PETSC_ARCH=arch-linux2-c-debug
	SLEPC_ARCH=arch-linux2-c-debug
	BOOST_ROOT=/home/evan/local/boost_1_63_0
endif

include $(SLEPC_DIR)/lib/slepc/conf/slepc_common
CFLAGS = -Wall -std=c++11 -fPIC

#GELLMANN
ifeq ($(COMP),Gellmann)
CFLAGS+=-L/home/ecurtin2/local/lib/ -I/home/ecurtin2/local/include/ -lopenblas -larmadillo
LINKEROPTS = -L/home/ecurtin2/local/lib/ -larmadillo
endif

debug1D: CFLAGS += -g -o0 -Wall -gdwarf-3 -D NDIM=1
debug1D: EXE = HFSdebug1D
debug1D: BUILDDIR = build/debug1D
debug1D: $(TARGET)

profile1D: CFLAGS += -pg -O3 -Wall -D ARMA_NO_DEBUG -D NDEBUG -D PROFILE -D NDIM=1
profile1D: EXE = HFSprofile1D
profile1D: BUILDDIR = build/profile1D
profile1D:  $(TARGET)

release1D: CFLAGS += -O3 -Wall -D ARMA_NO_DEBUG -D NDEBUG -D RELEASE -D NDIM=1
release1D: EXE = HFSrelease1D
release1D: BUILDDIR = build/release1D
release1D: $(TARGET)

debug2D: CFLAGS += -g -o0 -Wall -gdwarf-3 -D NDIM=2
debug2D: EXE = HFSdebug2D
debug2D: BUILDDIR = build/debug2D
debug2D: $(TARGET)

profile2D: CFLAGS += -pg -O3 -Wall -D ARMA_NO_DEBUG -D NDEBUG -D PROFILE -D NDIM=2
profile2D: EXE = HFSprofile2D
profile2D: BUILDDIR = build/profile2D
profile2D:  $(TARGET)

release2D: CFLAGS += -O3 -Wall -D ARMA_NO_DEBUG -D NDEBUG -D RELEASE -D NDIM=2
release2D: EXE = HFSrelease2D
release2D: BUILDDIR = build/release2D
release2D: $(TARGET)

debug3D: CFLAGS += -g -o0 -Wall -gdwarf-3 -D NDIM=3
debug3D: EXE = HFSdebug3D
debug3D: BUILDDIR = build/debug3D
debug3D: $(TARGET)

profile3D: CFLAGS += -pg -O3 -Wall -D ARMA_NO_DEBUG -D NDEBUG -D PROFILE -D NDIM=3
profile3D: EXE = HFSprofile3D
profile3D: BUILDDIR = build/profile3D
profile3D:  $(TARGET)

release3D: CFLAGS += -O3 -Wall -D ARMA_NO_DEBUG -D NDEBUG -D RELEASE -D NDIM=3
release3D: EXE = HFSrelease3D
release3D: BUILDDIR = build/release3D
release3D: $(TARGET)

.PHONY: depend clean

$(TARGET): $(OBJECTS)
	@echo " SLEPC_EPS_LIB = ${SLEPC_EPS_LIB}"
	@echo " Linking..."
	@mkdir -p bin
	@cd $(BUILDDIR) && echo " $(CC) $^ -o ../../bin/$(EXE) $(LIB) ${SLEPC_EPS_LIB}"; $(CC) $^ -o ../../bin/$(EXE) $(LIB) ${SLEPC_EPS_LIB} $(LINKEROPTS)

%.o: $(SRCDIR)/%.$(SRCEXT)
	@echo "Building $@..."
	@mkdir -p $(BUILDDIR)
	@cd $(BUILDDIR) && echo " $(CC) $(CFLAGS) $(INC) -c -o $@ ../../$<"; $(CC) $(CFLAGS) $(INC) -c -o $@ ../../$<

clean::
	@echo " Cleaning..."
	@rm -rf build/
	@rm -rf bin/

depend: $(SRCS)
	makedepend $(INCLUDES) $^

