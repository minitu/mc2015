
CXX=g++
CXXFLAGS=-Wall


LIBS = -lrt -lm -lOpenCL
LDFLAGS = ${LIBS}


all: kmeans

.PHONY: all clean

kmeans: kmeans.o
	${CXX} $^ -o $@ ${LDFLAGS}

run:
	./gen_data.py centroid 16 centroid.point
	./gen_data.py data 1048576 data.point 16
	thorq --add --mode single --device gpu ./kmeans centroid.point data.point result.class final_centroid.point 1024
	
data:
	./gen_data.py centroid 16 centroid.point
	./gen_data.py data 1048576 data.point 16
	
test:
	thorq --add --mode single --device gpu --timeout 100 ./kmeans centroid.point data.point result.class final_centroid.point 1024

graph:
	./plot_data.py result final_centroid.point data.point result.class result.png

clear:
	rm -f task*

clean:
	rm -f kmeans kmeans.o centroid.point data.point final_centroid.point result.class task* result*.png
