#!/bin/bash

SPC_PATH=/home/tom/data_input/total_nitrogen/spc/spc.csv
WET_PATH=/home/tom/data_input/total_nitrogen/wetchem/wetchem.csv
SPLITS_DIR=/home/tom/data_input/total_nitrogen/splits
BASE_DIR=/home/tom/data_input/total_nitrogen/
SNAPSHOT_DIR=/home/tom/data_input/total_nitrogen/snaphots
chem="total_nitrogen"
#python3 /home/tom/DSML125/dl/optimize.py --trials 30 --chemical ${chem} --opts 100 --epochs 200  --spc-path ${SPC_PATH} --wet-path ${WET_PATH} --splits-dir ${SPLITS_DIR} --base-dir ${BASE_DIR} --snapshot-dir ${SNAPSHOT_DIR}
# python3 /home/tom/DSML125/dl/evaluate.py --chemical ${chem} --spc-path ${SPC_PATH} --wet-path ${WET_PATH} --splits-dir ${SPLITS_DIR} --base-dir ${BASE_DIR} --snapshot-dir ${SNAPSHOT_DIR}
# mkdir home/tom/${chem}
python3 /home/tom/DSML125/dl/compress.py --chemical ${chem} --base-dir ${BASE_DIR} --save-dir /home/tom/data_output
