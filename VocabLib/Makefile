# Makefile for VocabLib

MACHTYPE=$(shell uname -m)

OPTFLAGS=-g -O3 -ffast-math -Wall -mfpmath=sse -msse2 -funroll-loops -march=core2
OTHERFLAGS=-Wall -fopenmp -pthread

INCLUDE_PATH=-I../lib/ann_1.1/include/ANN -I../lib/ann_1.1_char/include/ANN \
	-I../lib/imagelib
LIBRARY_PATH=-L../lib/ann_1.1/lib -L../lib/ann_1.1_char/lib -L../lib/imagelib

CPPFLAGS=$(INCLUDE_PATH) $(OTHERFLAGS) $(OPTFLAGS)

LIB=libvocab.so

all: $(LIB)

$(LIB): $(OBJS)
	g++ -shared -fPIC *.cpp \
		$(OPTFLAGS) \
		$(OTHERFLAGS) \
		$(INCLUDE_PATH) \
		$(LIBRARY_PATH) \
		-Wl,-z,defs -Wl,--as-needed \
		-lANN -lANN_char -limage \
		-o $(LIB)

clean:
	rm -f *.o *~ $(LIB)
