#!/bin/bash

name="GPT4Scene-qwen2vl_full_sft_mark_32_3D_img512"
export PYTHONPATH=.

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
export CHUNKS=${#GPULIST[@]}

echo "Evaluating task in $CHUNKS GPUs"

BASE_ARGS=(
    "./evaluate/configs/eval_configs.py"
    "evaluate" "True"
    "model" "qwenvl"
    "batch_size" "4"
    "num_workers" "16"
    "model_path" "./model_outputs/${name}"
    "scene_anno" "./evaluate/annotation/selected_images_mark_3D_val_32.json"
    "save_interval" "2"
)

tasks=("scanqa" "sqa3d" "scan2cap" "scanrefer" "multi3dref")

for task in "${tasks[@]}"; do
    echo "Starting task: $task"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        echo "Launching chunk: $IDX for task: $task"
        ARGS=(
            "${BASE_ARGS[@]}"
            "val_tag" "$task"
            "output_dir" "eval_outputs/outputs_3D_mark/${name}/"
            "num_chunks" "$CHUNKS"
            "chunk_idx" "$IDX"
            "calculate_score_tag" "scanqa#sqa3d#scan2cap#scanrefer#multi3dref"
        )

        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python evaluate/infer.py "${ARGS[@]}" &
    done

    wait
    echo "Finished task: $task"
done

wait

CUDA_VISIBLE_DEVICES=0 python evaluate/calculate_scores.py "${ARGS[@]}"

wait