#!/bin/bash
# Restarts Tensorboard
kill -9 $(cat tensorboard.pid)
nohup tensorboard --logdir=results/ &> tensorboard.out &
echo $! > tensorboard.pid
