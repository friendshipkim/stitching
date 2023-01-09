#!/bin/bash

export WANDB_API_KEY=641959d1c0dbfc348e2e0b75279abe93425c6ec7
export WANDB_ENTITY=harvardml
export WANDB_PROJECT=stitching_nlp


bss=(32 64)
lrs=(1e-4 5e-5 3e-5)
max_epochs=(5 10)



count=0
for lr in "${lrs[@]}"; do
  for bs in "${bss[@]}"; do
    for max_epoch in "${max_epochs[@]}"; do
        echo "Running with lr=$lr, bs=$bs, max_epoch=$max_epoch on device=$count"
      
        CUDA_VISIBLE_DEVICES=$count bash scripts/train_mini-small.sh --learning_rate $lr --num_train_epochs=$max_epoch --per_device_train_batch_size=$bs &
      
        if [ $count -eq 3 ]; then
          wait
        fi
        count=$(((count+1)%4)) 
    done
  done
done

count=0
for lr in "${lrs[@]}"; do
  for bs in "${bss[@]}"; do
    for max_epoch in "${max_epochs[@]}"; do
        echo "Running with lr=$lr, bs=$bs, max_epoch=$max_epoch on device=$count"
      
        CUDA_VISIBLE_DEVICES=$count bash scripts/train_mini-mini-stitched.sh --learning_rate $lr --num_train_epochs=$max_epoch --per_device_train_batch_size=$bs &
      
        if [ $count -eq 3 ]; then
          wait
        fi
        count=$(((count+1)%4)) 
    done
  done
done