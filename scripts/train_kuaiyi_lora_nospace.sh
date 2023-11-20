export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

deepspeed --master_port=20002 toolbench/train/train_lora.py \
    --model_name_or_path ./kuaiyi_ckpt  \
    --data_path  data_kuaiyi/data_0921/out_cot_0920_nospace.json \
    --eval_data_path  data_kuaiyi/data_0921/out_cot_0920_nospace_val.json \
    --conv_template tool-llama-single-round \
    --bf16 True \
    --output_dir kuaiyi_lora_0921_nospace \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 8 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --source_model_max_length 2048 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed ds_configs/stage2.json \
    --report_to none
