#!/bin/bash

#echo "\$1=$1"
if [ "$1" = "" ];then
	echo "Please specify -np X"
else
	mpirun --hostfile hostfile -np $1 Dormoy input1.txt
fi
