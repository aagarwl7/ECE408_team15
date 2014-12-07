#!/bin/sh
cd /home/tomei2/ECE408_team15/
make clean
make
rm xyspin.*
submitjob xyspin $1 $2 $3
while [ ! -f xyspin.o* ]; do
		sleep 1
done