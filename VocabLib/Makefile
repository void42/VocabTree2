# Makefile for VocabLib

MACHTYPE=$(shell uname -m)

OPT_FLAGS=-g -O3 -ffast-math -Wall -mfpmath=sse -msse2 -funroll-loops -march=core2

INCL_PATH=-I../lib/ann_1.1/include/ANN -I../lib/ann_1.1_char/include/ANN \
	-I../lib/imagelib
LIB_PATH=-L../lib/ann_1.1/lib -L../lib/ann_1.1_char/lib -L../lib/imagelib
LIB_FLAGS=-Wl,-z,defs -Wl,--as-needed -lANN -lANN_char -limage

CC_FLAGS=-std=c++11 -Wall -pthread -fopenmp

LIB=libvocab.so

CPP_FILES := $(wildcard *.cpp)
OBJ_FILES := $(notdir $(CPP_FILES:.cpp=.o))

all: $(LIB)

%.o: %.cpp
	clang++ -fPIC \
		$(CC_FLAGS) \
		$(OPT_FLAGS) \
		$(INCL_PATH) \
		$(LIB_PATH) \
		-c -o $@ $<

$(LIB): $(OBJ_FILES)
	clang++ -shared -fPIC \
		$(CC_FLAGS) \
		$(OPT_FLAGS) \
		$(INCL_PATH) \
		$(LIB_PATH) \
		-o $@ $^ \
		$(LIB_FLAGS)

clean:
	rm -f *.o *~ $(LIB)
