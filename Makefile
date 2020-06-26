CFLAGS= -Xcompiler -fopenmp
INCS=
# use the following line to "Purify" the code
#CC=purify gcc
CC=nvcc
SRCS=src/BaumWelch.c src/viterbi.c src/hmmutils.c \
  src/emutils.c src/nrutil.c src/esthmm.c src/hmmrand.c \
  src/logmath.c src/fwd_bwd.c src/fileutils.c

all :	TRACE

TRACE: src/esthmm.o src/BaumWelch.o src/nrutil.o src/hmmutils.o \
    src/emutils.o src/logmath.o src/fwd_bwd.o src/viterbi.o src/fileutils.o
	 $(CC) -Xcompiler -fopenmp --use_fast_math -gencode arch=compute_50,code=sm_50 -c src/cu_fwd_bwd.cu -o src/cu_fwd_bwd.o
	 $(CC) -Xcompiler -fopenmp -o TRACE src/esthmm.o src/nrutil.o \
    src/emutils.o src/hmmutils.o src/logmath.o src/fwd_bwd.o \
    src/BaumWelch.o src/viterbi.o src/fileutils.o src/cu_fwd_bwd.o -lgsl -lgslcblas -lm

clean:
	rm src/*.o 
	rm TRACE
# DO NOT DELETE THIS LINE -- make depend depends on it.

