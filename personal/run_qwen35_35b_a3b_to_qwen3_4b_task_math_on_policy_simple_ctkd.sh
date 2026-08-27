set -e
set -x

# EXP_DIR: 模型与数据集所在根目录（可通过环境变量注入，默认 /root）
#   ${EXP_DIR}/Qwen3-4B   学生模型
#   ${EXP_DIR}/Qwen3.5-9B  教师模型
#   ${EXP_DIR}/dapo-math-17k    训练数据集
export EXP_DIR=${EXP_DIR:-/root}

# Start ray before first running
# ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# ============ TrainingArguments ============
OPTS=""
OPTS+=" --num_nodes 1"
OPTS+=" --num_gpus_per_node 8"
OPTS+=" --backend fsdp2"
OPTS+=" --train_batch_size 128"
OPTS+=" --micro_train_batch_size 2"
OPTS+=" --learning_rate 2e-5"
OPTS+=" --lr_warmup_ratio 0.05"
OPTS+=" --num_epochs 1"
OPTS+=" --eval_steps 5"
OPTS+=" --save_path ./output/qwen3_5_9b_to_qwen3_4b"
OPTS+=" --bf16 True"
OPTS+=" --gradient_checkpointing True"
OPTS+=" --enable_sleep True"

# ============ ModelArguments ============
OPTS+=" --student_name_or_path ${EXP_DIR}/Qwen3-4B"
OPTS+=" --teacher_name_or_path ${EXP_DIR}/Qwen3.5-9B"
OPTS+=" --enable_thinking True"

# ============ RolloutArguments ============
OPTS+=" --rollout_batch_size 128"
OPTS+=" --rollout_num_engines 8"
OPTS+=" --rollout_tp_size 1"
OPTS+=" --rollout_mem_fraction_static 0.6"
OPTS+=" --n_samples_per_prompt 1"
OPTS+=" --generate_max_len 8192"
OPTS+=" --eval_n_samples_per_prompt 8"
OPTS+=" --eval_generate_max_len 16384"
OPTS+=" --eval_top_p 0.7"

# ============ DataArguments ============
OPTS+=" --train_dataset_path ${EXP_DIR}/dapo-math-17k"
OPTS+=" --eval_dataset_path ${EXP_DIR}/aime-2024"
OPTS+=" --eval_split train"
OPTS+=" --max_len 4096"
OPTS+=" --input_key prompt"
OPTS+=" --label_key label"
OPTS+=" --custom_eval_fn examples/evaluation/math_dapo_utils.py"
OPTS+=" --custom_reward_fn examples/evaluation/math_dapo_utils.py"
OPTS+=" --apply_chat_template True"
OPTS+=" --preprocess_num_workers 32"
OPTS+=" --packing_samples True"

# ============ DistillationArguments ============
OPTS+=" --kd_ratio 1.0"
OPTS+=" --kd_loss_fn rkl"
OPTS+=" --kd_algorithm simple_ctkd"
OPTS+=" --teacher_dp_size 2"
OPTS+=" --teacher_tp_size 4"
OPTS+=" --teacher_mem_fraction_static 0.6"

# ============ LoggingArguments ============
OPTS+=" --logging_steps 1"
OPTS+=" --use_wandb True"
OPTS+=" --sync_swanlab True"
OPTS+=" --wandb_project KDFlow"
OPTS+=" --wandb_group cross_tokenizer_kd_onpolicy"
OPTS+=" --wandb_run_name qwen3_5_9b_to_qwen3_4b_simple_ctkd"
OPTS+=" --wandb_mode offline"
OPTS+=" --wandb_dir ./output"

python -m kdflow.cli.train_kd_on_policy $OPTS
