#!/bin/bash
set -x -e
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

name="qwen2vl_7b_full_sft_mark_32_3D_img512"

export NNODES=1
export num_gpus=8
export WANDB_DISABLED=true
export full_batch_size=16
export batch_size=1
export gradient_accumulation_steps=$[$full_batch_size/($batch_size*$num_gpus*$NNODES)]
export CPUS_PER_TASK=20
export MASTER_PORT=$((RANDOM % 101 + 29400))

export output_dir=model_outputs/${name}/
export model_name_or_path=./ckpts/models--Qwen--Qwen2-VL-7B-Instruct/
export tokenized_path=tokenizer/qwen2vl_full_sft_mark_32_3D_img512/

## slurm
export PARTITION=mllm
export JOB_NAME=gpt4scene
export QUOTA_TYPE=reserved

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${num_gpus} \
    --time=2-00:00:00 \
    --quota=$QUOTA_TYPE \
    --nodes=${NNODES} \
    --ntasks-per-node=1 \
    --cpus-per-task=${CPUS_PER_TASK} \
    bash -c 'torchrun \
    --nnodes $NNODES \
    --nproc_per_node ${num_gpus:-1} \
    --node_rank="${SLURM_NODEID}" \
    --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n1) \
    --master_port=$MASTER_PORT \
    src/train.py \
    --tokenized_path $tokenized_path \
    --model_name_or_path $model_name_or_path \
    --do_train true \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --finetuning_type full \
    --dataset sharegpt_data_chat_scene_32_images_3D_mark \
    --image_resolution 512 \
    --template qwen2_vl \
    --cutoff_len 32768 \
    --overwrite_cache true \
    --preprocessing_num_workers 16 \
    --output_dir $output_dir \
    --num_train_epochs 1.0 \
    --logging_steps 10 \
    --save_steps 4000 \
    --save_total_limit 1 \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate 5.0e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --flash_attn fa2 \
    --report_to none'