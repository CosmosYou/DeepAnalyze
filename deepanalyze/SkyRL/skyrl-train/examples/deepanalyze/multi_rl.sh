export CUDA_VISIBLE_DEVICES=0,1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export RAY_memory_monitor_refresh_ms=0
export API_KEY="sk-ea09308a837b4411a6fbc468599edfdb"
export BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export MODEL_NAME="qwen-plus"

NUM_GPUS=2
# 请确保以下路径为绝对路径
MODEL_COLDSTART_PATH="/root/ckpts/deepanalyze_8b_lora_2gpu"
FINAL_MODEL_PATH="/root/ckpts/deepanalyze_8b_lora_2gpu"
DATA_DIR="/usr/local/data"
INFERENCE_BACKEND="vllm"

# bash examples/deepanalyze/multi_rl.sh
# 切换到 skyrl-train 目录 (假设您当前不在该目录)
# cd /usr/local/DeepAnalyze/deepanalyze/SkyRL/skyrl-train

python -m examples.deepanalyze.main_deepanalyze \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.epochs=1 \
    data.train_data="[\"${DATA_DIR}/report_rl_no_searchfunction.parquet\"]" \
    trainer.policy.model.path="${MODEL_COLDSTART_PATH}" \
    trainer.placement.colocate_all=true \
    trainer.strategy="fsdp2" \
    trainer.policy.fsdp_config.cpu_offload=true \
    trainer.ref.fsdp_config.cpu_offload=true \
    trainer.placement.policy_num_gpus_per_node=${NUM_GPUS} \
    trainer.placement.ref_num_gpus_per_node=${NUM_GPUS} \
    generator.num_inference_engines=1 \
    generator.inference_engine_tensor_parallel_size=2 \
    generator.gpu_memory_utilization=0.5 \
    '+generator.engine_init_kwargs.max_model_len=4096' \
    trainer.train_batch_size=32 \
    trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.max_prompt_length=4000 \
    generator.max_input_length=12288 \
    generator.sampling_params.max_generate_length=12288 \
    trainer.policy.optimizer_config.lr=5e-7 \
    trainer.policy_mini_batch_size=4 \
    trainer.algorithm.use_kl_loss=false \
    generator.backend="${INFERENCE_BACKEND}" \
    generator.run_engines_locally=true \
    generator.weight_sync_backend="nccl" \
    generator.async_engine=true \
    generator.batched=true \
    generator.use_conversation_multi_turn=false \
    generator.n_samples_per_prompt=5 \
    generator.max_turns=30 \
    generator.sampling_params.temperature=0.0 \
    generator.sampling_params.top_p=0.95 \
    +generator.sampling_params.stop_token_ids="[151676,151645]" \
    environment.env_class="deepanalyze" \
    environment.skyrl_gym.deepanalyze.workspace="${DATA_DIR}/" \
    environment.skyrl_gym.deepanalyze.api_key="${API_KEY}" \
    environment.skyrl_gym.deepanalyze.base_url="${BASE_URL}" \
    environment.skyrl_gym.deepanalyze.llm_judgement_model="${MODEL_NAME}" \
    trainer.logger="[\"console\",\"tensorboard\"]" \
    trainer.project_name="deepanalyze" \
    trainer.run_name="deepanalyze_run" \
    trainer.resume_mode=null \
    trainer.ckpt_path="${FINAL_MODEL_PATH}/ckpt" \
    trainer.export_path="${FINAL_MODEL_PATH}/export" \
    trainer.eval_batch_size=4 \
    trainer.eval_before_train=false \
    trainer.eval_interval=-1 \
    trainer.hf_save_interval=1 \
    trainer.ckpt_interval=1 \
