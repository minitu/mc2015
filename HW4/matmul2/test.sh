#!/bin/bash
for i in {1..1024}
do
	echo "Executing mat_mul -s $i"
	./mat_mul -s $i >> result.txt
done
