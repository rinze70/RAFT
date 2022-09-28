#!/bin/bash
mkdir -p checkpoints
python3 -u train.py --name raft-chairs --stage chairs --validation chairs --gpus 0 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001