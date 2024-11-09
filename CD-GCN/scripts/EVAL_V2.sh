#!/bin/bash

RECORD=joint
WORKDIR=output/train/$RECORD
MODELNAME=output/train/$RECORD

CONFIG=./config/uav-cross-subjectv2/test.yaml

WEIGHTS=/home/coop/yuxin/CDresults/joint/runs/joint2-42-12814.pt

BATCH_SIZE=128

python3 main.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --weights $WEIGHTS
