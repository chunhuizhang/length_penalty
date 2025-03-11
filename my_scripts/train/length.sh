set -x

ray stop

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
ray start --head --node-ip-address 0.0.0.0 --num-gpus $NUM_GPUS --ray-debugger-external

export VLLM_ATTENTION_BACKEND=XFORMERS
MODEL_PATH=agentica-org/DeepScaleR-1.5B-Preview

LR=1e-6
ROLLOUT_N=8
TRAIN_BZ=32

# ALGO
BETA=0.5

# 7391/32 = 232 steps

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files=$HOME/data/length/MATH_Hard.parquet \
    data.val_files=$HOME/data/length/test_aime_reasoning.parquet \
    data.train_batch_size=$TRAIN_BZ \
    data.max_prompt_length=2048 \
    data.max_response_length=32000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=80 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_length' \
    trainer.experiment_name='length_penalty_'$LR'_'$ROLLOUT_N'_'$TRAIN_BZ \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=4 $@