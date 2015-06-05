
CXX=mpicc
CXXFLAGS=-Wall


LIBS = -lrt
LDFLAGS = ${LIBS}


all: seq mpi

.PHONY: all seq mpi clean test


seq: kmeans_seq

kmeans_seq: kmeans_seq.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS}


mpi: kmeans_mpi

kmeans_mpi: kmeans_mpi.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS}

run_all:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 65536 data.point 64
	thorq --add kmeans_seq centroid.point data.point result_seq.class final_centroid_seq.point 1024
	thorq --add kmeans_mpi centroid.point data.point result_mpi.class final_centroid_mpi.point 1024

run_seq:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 65536 data.point 64
	thorq --add kmeans_seq centroid.point data.point result_seq.class final_centroid_seq.point 1024

run:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 65536 data.point 64
	thorq --add kmeans_mpi centroid.point data.point result_mpi.class final_centroid_mpi.point 1024
	

clean:
	rm -f kmeans_seq kmeans_mpi kmeans_main.o kmeans_seq.o kmeans_mpi.o centroid.point data.point final_centroid*.point result*.class task*
