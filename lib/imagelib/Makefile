# Makefile for imagelib

MACHTYPE=$(shell uname -m)

GCC			= g++

CC=gcc
OPTFLAGS=-g2
# OPTFLAGS=-O3
OTHERFLAGS=-Wall

IMAGELIB_OBJS= qsort.o util.o

CFLAGS=$(OTHERFLAGS) $(OPTFLAGS)

IMAGELIB=libimage.so

all: $(IMAGELIB)

$(IMAGELIB):
	gcc -shared -fPIC $(OTHERFLAGS) *.c -o $(IMAGELIB)

clean:
	rm -f *.o *~ $(IMAGELIB)
