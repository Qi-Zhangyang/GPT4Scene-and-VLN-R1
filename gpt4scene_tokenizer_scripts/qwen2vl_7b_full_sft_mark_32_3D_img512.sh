#!/bin/bash
set -x -e
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

name="qwen2vl_7b_full_sft_mark_32_3D_img512"

export full_batch_size=2
export batch_size=1
export gradient_accumulation_steps=$[$full_batch_size/2] # only for tokenizer, so Here, I just randomly filled in "2".

export output_dir=outputs/vlnce/${name}/
export model_name_or_path=ckpts/models--Qwen--Qwen2-VL-7B-Instruct
export tokenized_path=tokenizers/${name}/


python src/train.py \
    --tokenized_path $tokenized_path \
    --model_name_or_path $model_name_or_path \
    --do_train true \
    --stage sft \
    --finetuning_type full \
    --dataset sharegpt_data_chat_scene_32_images_3D_mark \
    --image_resolution 512 \
    --template qwen2_vl \
    --cutoff_len 32768 \
    --overwrite_cache true \
    --preprocessing_batch_size 256 \
    --preprocessing_num_workers 256 \
    --output_dir $output_dir \
    --num_train_epochs 2.0 \
    --logging_steps 10 \
    --save_steps 18750 \
    --save_total_limit 1 \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate 5.0e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --flash_attn fa2 \
    --report_to none