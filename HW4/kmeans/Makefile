
CXX=g++
CXXFLAGS=-Wall


LIBS = -lrt
LDFLAGS = ${LIBS}


all: seq pthread

.PHONY: all seq pthread clean test


seq: kmeans_seq

kmeans_seq: kmeans_seq.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS}


pthread: kmeans_pthread

kmeans_pthread: kmeans_pthread.o kmeans_main.o
	${CXX} $^ -o $@ ${LDFLAGS} -lpthread -lm

run_all:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 65536 data.point 64
	thorq --add kmeans_seq centroid.point data.point result_seq.class final_centroid_seq.point 1024
	thorq --add kmeans_pthread centroid.point data.point result_pthread.class final_centroid_pthread.point 1024

run_seq:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 65536 data.point 64
	thorq --add kmeans_seq centroid.point data.point result_seq.class final_centroid_seq.point 1024

run:
	./gen_data.py centroid 64 centroid.point
	./gen_data.py data 65536 data.point 64
	thorq --add kmeans_pthread centroid.point data.point result_pthread.class final_centroid_pthread.point 1024
	

clean:
	rm -f kmeans_seq kmeans_pthread kmeans_main.o kmeans_seq.o kmeans_pthread.o centroid.point data.point final_centroid.point final_centroid_seq.point final_centroid_pthread.point result.class result_seq.class result_pthread.class task*
