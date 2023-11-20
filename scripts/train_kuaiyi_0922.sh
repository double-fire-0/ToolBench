export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=6,7


torchrun --nproc_per_node=2 --master_port=20001 toolbench/train/train_mem.py \
    --do_train False  \
    --do_predict True \
    --model_name_or_path ./kuaiyi_ckpt  \
    --data_path  data_kuaiyi/data_0921/out_cot_0922_con.json \
    --eval_data_path  data_kuaiyi/data_0921/out_cot_0922_con_val.json \
    --conv_template tool-llama-single-round \
    --bf16 True \
    --output_dir ./kuaiyi_0922 \
    --num_train_epochs 4 \
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
    --logging_dir ./kuaiyi_0922/logs \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --source_model_max_length 2048 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to tensorboard \
    --eval_accumulation_steps 2 \
