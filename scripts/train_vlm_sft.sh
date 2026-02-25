#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

# DeepSpeed configuration
deepspeed=./scripts/zero2.json

# Model configuration
model_path=Qwen/Qwen2.5-VL-3B-Instruct/
geometry_encoder_type=spa3r
geometry_encoder_path=spa3r.ckpt

# Training hyperparameters
lr=1e-5
batch_size=1
grad_accum_steps=8

# Training entry point
entry_file=spa3_vlm/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=vsi_590k

# Output configuration
output_dir=outputs/spa3_vlm

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${model_path}" \
    --dataset_use ${datasets} \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels $((576*28*28)) \
    --video_max_frames 8 \
    --video_max_pixels $((1664*28*28)) \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 50 \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --group_by_modality_length true \
    --use_geometry_encoder true \
    --geometry_encoder_type ${geometry_encoder_type} \
    --geometry_encoder_path ${geometry_encoder_path}"

# Launch training
echo ${args}
torchrun --nproc_per_node=${NPROC_PER_NODE} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${MASTER_PORT} \
            ${entry_file} ${args}
