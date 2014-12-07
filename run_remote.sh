#!/bin/sh
#rm xyspin.*
scp -r ./*.cu ./*.sh ./*.h ./Makefile tomei2@gem.csl.illinois.edu:/home/tomei2/ECE408_team15/
ssh tomei2@gem.csl.illinois.edu /home/tomei2/ECE408_team15/run_xyspin.sh $1 $2 $3
scp tomei2@gem.csl.illinois.edu:/home/tomei2/ECE408_team15/xyspin.* .
#cat xyspin.o* | more