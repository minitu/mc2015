
CXX=g++
CXXFLAGS=-O2 -Wall -fopenmp


LIBS = -lrt
LDFLAGS = ${LIBS}


all: seq omp

.PHONY: all seq omp clean test


seq: kmeans_seq

kmeans_seq: kmeans_seq.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS}


omp: kmeans_omp

kmeans_omp: kmeans_omp.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS} -lgomp

run_all:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 65536 data.point 64
	thorq --add kmeans_seq centroid.point data.point result_seq.class final_centroid_seq.point 1024
	thorq --add kmeans_omp centroid.point data.point result_omp.class final_centroid_omp.point 1024

run_seq:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 65536 data.point 64
	thorq --add kmeans_seq centroid.point data.point result_seq.class final_centroid_seq.point 1024

run:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 65536 data.point 64
	thorq --add kmeans_omp centroid.point data.point result_omp.class final_centroid_omp.point 1024

data:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 65536 data.point 64

local_seq:
	./kmeans_seq centroid.point data.point result_seq.class final_centroid_seq.point 1024

local_omp:
	./kmeans_omp centroid.point data.point result_omp.class final_centroid_omp.point 1024

graph_seq:
	./plot_data.py result final_centroid_seq.point data.point result_seq.class result_seq.png

graph_omp:
	./plot_data.py result final_centroid_omp.point data.point result_omp.class result_omp.png

clean:
	rm -f kmeans_seq kmeans_omp kmeans_main.o kmeans_seq.o kmeans_omp.o centroid.point data.point final_centroid*.point result*.class task*
