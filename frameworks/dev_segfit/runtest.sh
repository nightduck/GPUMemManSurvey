#!/bin/bash
sudo -v

if [ -z "$1" ]
then
  filename="filename"
else 
  filename=$1
fi

nvcc -o segfit main.cu -lcuda -lnvToolsExt
sudo nsys profile --cuda-memory-usage=true --output=$filename.nsys-rep ./segfit
nsys stats --report cuda_api_trace --format csv ./$filename.nsys-rep > $filename\_api.csv

# Comment out this line for the vanilla run
nsys stats --report nvtx_pushpop_trace --format csv ./$filename.nsys-rep > $filename\_nvtx.csv