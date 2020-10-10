#!/usr/bin/env bash

# export ZHEN_DIR='wmt16-ro-en' # Download instructions above
export ZHEN_DIR='/data00/wuwei.ai/data/translation/shenjian_zh_en_tfmr' # Download instructions above
# export WANDB_PROJECT="MT" # optional
export MAX_LEN=64
export BS=32
export PYTHONPATH="../":"${PYTHONPATH}"

python finetune.py \
    --num_train_epochs 6 --src_lang zh_CN --tgt_lang en_XX \
    --learning_rate=3e-5 \
    --fp16 \
    --do_train \
    --val_check_interval=0.25 \
    --adam_eps 1e-06 \
    --data_dir $ZHEN_DIR \
    --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN --eval_max_gen_length $MAX_LEN \
    --train_batch_size=$BS --eval_batch_size=1 \
    --task translation \
    --warmup_steps 500 \
    --freeze_embeds \
    --model_name_or_path=facebook/mbart-large-cc25 \
    --output_dir zhen_finetune_04 \
    --label_smoothing 0.1 \
    --fp16_opt_level=O1 \
    --logger_name wandb \
    --eval_beams=1 \
    --gpus=1 \
    "$@"
    # --do_predict \
