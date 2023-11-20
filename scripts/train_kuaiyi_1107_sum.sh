export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


torchrun --nproc_per_node=8 --master_port=20001 toolbench/train/train_mem.py \
    --do_train True  \
    --do_predict False \
    --model_name_or_path ./kuaiyi_ckpt  \
    --data_path  /hetu_group/linhezheng/database/ai_acount/ai_planner/datas/train_kuaiyi_data/train_sum_all_data.json \
    --eval_data_path  data_kuaiyi/data_0921/out_sum_1009_con_val.json \
    --conv_template tool-llama-single-round \
    --bf16 True \
    --output_dir ./kuaiyi_sum_1107 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 8 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_dir ./kuaiyi_1107/logs \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --source_model_max_length 2048 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none \
    --eval_accumulation_steps 2 \
    --shuffle_train_data True \