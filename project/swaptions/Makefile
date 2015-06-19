DEF = 
LIBS = -lrt

EXEC = swaptions 

ifdef version
  ifeq "$(version)" "pthreads" 
    DEF := $(DEF) -DENABLE_THREADS
    CXXFLAGS := $(CXXFLAGS) -pthread
  endif
  ifeq "$(version)" "tbb"
    DEF := $(DEF) -DENABLE_THREADS -DTBB_VERSION
    LIBS := $(LIBS) -ltbb
  endif
  ifeq "$(version)" "cpu"
    DEF := $(DEF) -DUSE_CPU
    LIBS := $(LIBS) -lOpenCL
  endif
  ifeq "$(version)" "gpu"
    DEF := $(DEF) -DUSE_GPU
    LIBS := $(LIBS) -lOpenCL
  endif
  ifeq "$(version)" "mpi"
    CXX := mpicxx
    DEF := $(DEF) -DUSE_MPI
    LIBS := $(LIBS) -lOpenCL
  endif
  ifeq "$(version)" "snucl"
    CXX := mpicxx
    DEF := $(DEF) -DUSE_SNUCL
    INCLUDE := $(INCLUDE) -I$(SNUCLROOT)/inc -I$(SNUCLROOT)/common.mk
    LIBS := $(LIBS) -L$(SNUCLROOT)/lib -lsnucl_cluster
  endif
endif

OBJS= CumNormalInv.o MaxFunction.o RanUnif.o nr_routines.o icdf.o \
	HJM_SimPath_Forward_Blocking.o HJM.o HJM_Swaption_Blocking.o  \
	HJM_Securities.o

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(DEF) $(OBJS) $(INCLUDE) $(LIBS) -o $(EXEC)

.cpp.o:
	$(CXX) $(CXXFLAGS) $(DEF) $(INCLUDE) $(LIBS) -c $*.cpp -o $*.o

.c.o:
	$(CXX) $(CXXFLAGS) $(DEF) $(INCLUDE) $(LIBS) -c $*.c -o $*.o

clean:
	rm -f $(OBJS) $(EXEC)

