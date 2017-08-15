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

all: 
	@make NDIM=1 MODE=$(MODE)
	@make NDIM=2 MODE=$(MODE)
	@make NDIM=3 MODE=$(MODE)

SOURCES = $(wildcard $(SRCDIR)/*$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/$(MODE)/$(NDIM)d/%,$(SOURCES:.$(SRCEXT)=.o))

LIB = -larmadillo -L/usr/lib

MANSEC = EPS
CFLAGS = -Wall -std=c++11 -fPIC -D NDIM=$(NDIM)

ifeq ($(MODE),release)
CFLAGS += -O3 -Wall -D ARMA_NO_DEBUG -D NDEBUG -D RELEASE
else
CFLAGS += -g -o0 -Wall -gdwarf-3
endif

INC =  -I$(SLEPC_DIR)/include 
INC += -I$(PETSC_DIR)/include 
INC += -I$(BOOST_ROOT)
INC += -I$(ARMA_INC)
include $(SLEPC_DIR)/lib/slepc/conf/slepc_common


$(BINDIR)/$(MODE)/$(NDIM)d/$(TARGET): $(OBJECTS)
	@echo " Linking..."
	@mkdir -p $(BINDIR)/$(MODE)/$(NDIM)d
	@echo "$(CXX) $^ -o $@"
	@$(CXX) $^ -o $@ $(LIB) $(INC) ${SLEPC_EPS_LIB}

$(OBJECTS): $(BUILDDIR)/$(MODE)/$(NDIM)d/%.o : $(SRCDIR)/%.cpp
	@echo "CXX -c $< -o $@"
	@mkdir -p $(BUILDDIR)/$(MODE)/$(NDIM)d
	@$(CXX) $(CFLAGS) -c $< $(INC) $(LIB) -o $@

.PHONY: clean print

print:
	$(info $$SOURCES is [${SOURCES}])
	$(info $$OBJECTS is [${OBJECTS}])
	$(info $$BINDIR is [${BINDIR}])
	$(info $$MODE is [${MODE}])
	$(info $$TARGET is [${TARGET}])
	$(info $$CFLAGS is [${CFLAGS}])
	$(info $$NDIM is [${NDIM}])

clean:
	@echo " Cleaning..."
	@rm -rf build/
	@rm -rf bin/
