export CUDA_VISIBLE_DEVICES="4,5,6,7"
torchrun --nproc_per_node=4 --master_port=3389 train.py \
    --model_name_or_path "models/bigscience_bloom-3b" \
    --train_dataset ./alpaca_data.json \
    --eval_dataset ./alpaca_data_eval.json \
    --fp16 True \
    --output_dir ./checkpoints/bloom-3b \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config fsdp_config_bloom.json \
    --gradient_checkpointing False

# --deepspeed ds_config.json
# --bf16 True \
# --tf32 True