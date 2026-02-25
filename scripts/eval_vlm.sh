export LMMS_EVAL_PLUGINS=spa3_vlm.lmms_eval

model=spa3_vlm
model_path=outputs/spa3_vlm

accelerate launch --num_processes=8 -m lmms_eval \
    --model "${model}" \
    --model_args pretrained="${model_path}",attn_implementation=flash_attention_2 \
    --tasks vsibench \
    --batch_size 1
