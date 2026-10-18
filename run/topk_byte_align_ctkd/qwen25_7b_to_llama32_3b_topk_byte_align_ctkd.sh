set -e
set -x
set -o pipefail   


# ============ Logging Path ============
TRAIN_NAME="qwen25_7b_to_llama32_3b_topk_byte_align_ctkd_16"
SAVE_PATH="./outputs/${TRAIN_NAME}"
LOG_DIR="${SAVE_PATH}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
echo "Training log will be written to: ${LOG_FILE}"

export PYTHONPATH="/workspace/Cross-Tokenizer-OPD:${PYTHONPATH}"
# ============ TrainingArguments ============
OPTS=""
OPTS+=" --num_nodes 1"
OPTS+=" --num_gpus_per_node 8"
OPTS+=" --backend fsdp2"
OPTS+=" --train_batch_size 512"
OPTS+=" --micro_train_batch_size 4"
OPTS+=" --learning_rate 1e-6"
OPTS+=" --lr_scheduler constant"
OPTS+=" --num_epochs 10"
OPTS+=" --max_steps 100"
OPTS+=" --save_path ${SAVE_PATH}"
OPTS+=" --bf16 True"
OPTS+=" --gradient_checkpointing True"
OPTS+=" --enable_sleep True"

# ============ ModelArguments ============
OPTS+=" --student_name_or_path /workspace/models/Llama-3.2-3B-Instruct"
OPTS+=" --teacher_name_or_path /workspace/models/Qwen2.5-7B-Instruct"
OPTS+=" --enable_thinking False"

# ============ RolloutArguments ============
OPTS+=" --rollout_batch_size 512"
OPTS+=" --rollout_num_engines 8"
OPTS+=" --rollout_tp_size 1"
OPTS+=" --rollout_mem_fraction_static 0.6"
OPTS+=" --n_samples_per_prompt 1"
OPTS+=" --generate_max_len 8192"
OPTS+=" --temperature 1.0"
OPTS+=" --top_p 0.95"

# ============ DataArguments ============
OPTS+=" --train_dataset_path /workspace/data/Train/math_code_20k/train.parquet"
OPTS+=" --max_len 10240"
OPTS+=" --input_key prompt"
OPTS+=" --label_key reward_model.ground_truth"
OPTS+=" --apply_chat_template True"
OPTS+=" --preprocess_num_workers 32"
OPTS+=" --packing_samples True"
OPTS+=" --eval_math_dataset_path /workspace/data/math500/test.jsonl"
OPTS+=" --eval_math_max_tokens 8192"
OPTS+=" --eval_math_max_concurrent 256"
OPTS+=" --eval_math_timeout 2400"

# ============ DistillationArguments ============
OPTS+=" --kd_ratio 1.0"
OPTS+=" --kd_loss_fn rkl"
OPTS+=" --kd_algorithm topk_byte_align_ctkd"
OPTS+=" --ctkd_topk 16"
OPTS+=" --teacher_dp_size 4"
OPTS+=" --teacher_tp_size 2"
OPTS+=" --teacher_mem_fraction_static 0.6"
OPTS+=" --chunked_loss_size 2048"

# ============ LoggingArguments ============
OPTS+=" --logging_steps 1"
OPTS+=" --use_wandb False"
OPTS+=" --save_steps 10"
OPTS+=" --eval_math_n_samples 32"
OPTS+=" --eval_steps 10"

python -u -m kdflow.cli.train_kd_on_policy $OPTS 2>&1 | tee -a "${LOG_FILE}"
