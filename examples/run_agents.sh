#!/bin/bash

# 清空或创建日志文件
> wandb_agents.log

CUDA_VISIBLE_DEVICES=2 nohup wandb agent YOUR_ENTITY/YOUR_PROJECT/YOUR_SWEEP_ID_1 >> wandb_agents.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup wandb agent YOUR_ENTITY/YOUR_PROJECT/YOUR_SWEEP_ID_2 >> wandb_agents.log 2>&1 &

echo 'All sweep agents have been started in background.'
echo 'Check wandb_agents.log for output from all agents.'
