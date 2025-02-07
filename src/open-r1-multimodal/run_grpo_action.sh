cd src/open-r1-multimodal

export DEBUG_MODE="true"
export LOG_PATH="./debug_action_log_2b.txt"
export CUDA_VISIBLE_DEVICES="3,4,5,6,7"

RUN_NAME="Qwen2-VL-2B-GRPO-GUI-test"
PATH_TO_MODEL="/data/true_nas/zfs_share1/zyc/data/models/Qwen/Qwen2-VL-2B-Instruct"
PATH_TO_DATASET="/data/true_nas/zfs_share1/zyc/workspace/simpleRL-reason/prompt_dataset.jsonl"
OUTPUT_DIR="/data/true_nas/zfs_share1/zyc/expr"

torchrun --nproc_per_node="5" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_action.py \
    --output_dir $OUTPUT_DIR/$RUN_NAME \
    --model_name_or_path $PATH_TO_MODEL \
    --dataset_name $PATH_TO_DATASET \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --deepspeed /data/true_nas/zfs_share1/zyc/workspace/R1-V/src/open-r1-multimodal/local_scripts/zero3.json \
    --report_to tensorboard \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --max_pixels 50176 \
    2>&1 | tee -a "./${RUN_NAME}_training_log.txt"