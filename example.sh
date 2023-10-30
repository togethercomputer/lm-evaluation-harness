python -u main.py \
        --model tgi \
        --model_args model=${MODEL} \
        --tasks truthfulqa_gen,boolq,hellaswag,openbookqa,winogrande \
        --no_cache \
        --output_path /home/yucheng/lm-evaluation-harness/results/${SETTING}/${MODELSIZE} \
        --output_base_path /home/yucheng/lm-evaluation-harness/results/${SETTING}