#!/bin/bash
# Note that this orphans the process
# YOU are responsible for eventually stopping it via ps aux | grep cli and kill -9 PID
nohup ./cli.py train-aqn $* &> train.out &
