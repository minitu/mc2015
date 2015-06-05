
CXX=mpicxx
CXXFLAGS=-Wall
DEF=

LIBS = -lrt
LDFLAGS = ${LIBS}


all: seq mpi

.PHONY: all seq mpi clean test


seq: kmeans_seq

kmeans_seq: kmeans_seq.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS}


mpi: kmeans_mpi

kmeans_mpi: kmeans_mpi.o
	${CXX} $^ -o $@ ${LDFLAGS}

run_all:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 262144 data.point 64
	thorq --add kmeans_seq centroid.point data.point result_seq.class final_centroid_seq.point 1024
	thorq --add --mode mpi --nodes 4 --slots 16 kmeans_mpi centroid.point data.point result_mpi.class final_centroid_mpi.point 1024

run_seq:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 262144 data.point 64
	thorq --add kmeans_seq centroid.point data.point result_seq.class final_centroid_seq.point 1024

run:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 262144 data.point 64
	thorq --add --mode mpi --nodes 4 --slots 16 kmeans_mpi centroid.point data.point result_mpi.class final_centroid_mpi.point 1024

sd_mpi:
	thorq --add --mode mpi --nodes 4 --slots 16 kmeans_mpi centroid.point data.point result_mpi.class final_centroid_mpi.point 1024

graph_seq:
	./plot_data.py result final_centroid_seq.point data.point result_seq.class result_seq.png
	
graph_mpi:
	./plot_data.py result final_centroid_mpi.point data.point result_mpi.class result_mpi.png

clean:
	rm -f kmeans_seq kmeans_mpi kmeans_main.o kmeans_seq.o kmeans_mpi.o centroid.point data.point final_centroid*.point result*.class task*
