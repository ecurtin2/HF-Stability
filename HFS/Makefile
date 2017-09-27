#
# 'make'                  make NDIM=2 MODE=debug
# 'make NDIM=X MODE=Y     make version with ndim of 1, 2 or 3 and mode debug or release.
# 'make clean'            clean bin/ and build/
#

SRCDIR = src
BUILDDIR = build
BINDIR = bin
SRCEXT = cpp
TARGET = HFS
MODE = debug
NDIM = 2

debug2D(default): $(BINDIR)/$(MODE)/$(NDIM)d/$(TARGET)

test: all

	@echo "******************************************"
	@echo "*Testing all debug executables on 1 proc!*"
	@echo "******************************************"
	-$(MPI_RUN) -n 1 ./$(BINDIR)/debug/1d/$(TARGET)
	-$(MPI_RUN) -n 1 ./$(BINDIR)/debug/2d/$(TARGET)
	-$(MPI_RUN) -n 1 ./$(BINDIR)/debug/3d/$(TARGET) --Nk 7
    
	@echo "*******************************************"
	@echo "*Testing all debug executables on >1 proc!*"
	@echo "*******************************************"
	-$(MPI_RUN) -n 2 ./$(BINDIR)/debug/1d/$(TARGET)
	-$(MPI_RUN) -n 2 ./$(BINDIR)/debug/2d/$(TARGET)
	-$(MPI_RUN) -n 2 ./$(BINDIR)/debug/3d/$(TARGET) --Nk 7

all: 
	@make NDIM=1 MODE=$(MODE)
	@make NDIM=2 MODE=$(MODE)
	@make NDIM=3 MODE=$(MODE)

SOURCES = $(wildcard $(SRCDIR)/*$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/$(MODE)/$(NDIM)d/%,$(SOURCES:.$(SRCEXT)=.o))

LIB = -L$(LOCAL_LIB)

CFLAGS = -DARMA_DONT_USE_WRAPPER -Wall -std=c++11 -fPIC -D NDIM=$(NDIM)

ifeq ($(MODE),release)
CFLAGS += -pg -O3 -Wall -D ARMA_NO_DEBUG -D NDEBUG -D RELEASE
else
CFLAGS += -g -o0 -Wall -gdwarf-3
endif

INC =  -I$(SLEPC_DIR)/include 
INC +=  -I$(SLEPC_DIR)/$(SLEPC_ARCH)/include 

INC += -I$(ARMA_INC)
INC += $(CRAY_BOOST_INCLUDE_OPTS)
include $(SLEPC_DIR)/lib/slepc/conf/slepc_common
LIB += -L$(SLEPC_LIB_DIR) -lslepc

svar =-Wl,-rpath,/u/sciteam/curtin/local/SLEPc/slepc-3.7.4/installed-interlagos/lib -L/u/sciteam/curtin/local/SLEPc/slepc-3.7.4/installed-interlagos/lib -lslepc      -L/opt/cray/petsc/3.7.2.1/real/GNU/4.9/interlagos/lib  -lcraypetsc_gnu_49_real -L/opt/cray/libsci/13.2.0/GNU/4.9/x86_64/lib -lsci_gnu_mpi_mp -lsci_gnu_mp -lsuperlu -lHYPRE -lsundials_cvode -lsundials_cvodes -lsundials_ida -lsundials_idas -lsundials_kinsol -lsundials_nvecparallel -lsundials_nvecserial -lsci_gnu_mp -L/opt/cray/hdf5-parallel/1.8.16/GNU/4.9/lib -lhdf5_parallel -lz -ldl -lptscotch -lscotch -lptscotcherr -lscotcherr -ldl

$(BINDIR)/$(MODE)/$(NDIM)d/$(TARGET): $(OBJECTS)
	@echo " Linking..."
	@mkdir -p $(BINDIR)/$(MODE)/$(NDIM)d
#@echo "$(CXX) $^ -o $@"
	$(CXX) $(CFLAGS) $(svar) $(INC) $^ $(LIB) -o $@

$(OBJECTS): $(BUILDDIR)/$(MODE)/$(NDIM)d/%.o : $(SRCDIR)/%.cpp
#	@echo "CXX -c $< -o $@"
	@mkdir -p $(BUILDDIR)/$(MODE)/$(NDIM)d
	$(CXX) $(CFLAGS) -c $< $(INC) $(LIB) -o $@

.PHONY:: clean print

print:
	$(info $$SOURCES is [${SOURCES}])
	$(info $$OBJECTS is [${OBJECTS}])
	$(info $$BINDIR is [${BINDIR}])
	$(info $$MODE is [${MODE}])
	$(info $$TARGET is [${TARGET}])
	$(info $$CFLAGS is [${CFLAGS}])
	$(info $$NDIM is [${NDIM}])
	$(info $$BOOST_ROOT is [${BOOST_ROOT}])

.clean:
	@echo " Cleaning..."
	@rm -rf build/
	@rm -rf bin/
