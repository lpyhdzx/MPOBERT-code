set -x
base_dir=/home/liupeiyu/MPOBERT-code/src
check_point_dir=/mnt/liupeiyu/
data_dir_base=/home/liupeiyu/nlp_data/GLUE
train_data_dir=/home/liupeiyu/MPOBERT-code/toy_train_small
albert_checkpoint_path=/mnt/liupeiyu/checkpoint/albert-base-v2/pytorch_model.bin

function run_pretrain() {
  export CUDA_VISIBLE_DEVICES=$1
  COMMON_ARGS="--model_type=bert-mlm \
  --tokenizer_name=bert-large-uncased \
  --hidden_act=gelu \
  --hidden_size=1024 \
  --num_hidden_layers=24 \
  --num_attention_heads=16 \
  --intermediate_size=4096 \
  --hidden_dropout_prob=0.1 \
  --attention_probs_dropout_prob=0.1 \
  --attention_dropout_checkpoint=True \
  --gelu_checkpoint=True \
  --stochastic_mode=True \
  --encoder_ln_mode=post-ln \
  --zero=False \
  --lr=1e-3 \
  --train_batch_size=4096 \
  --train_micro_batch_size_per_gpu=32 \
  --lr_schedule=time \
  --curve=linear \
  --warmup_proportion=0.06 \
  --gradient_clipping=0.0 \
  --optimizer_type=adamw \
  --weight_decay=0.01 \
  --adam_beta1=0.9 \
  --adam_beta2=0.98 \
  --adam_eps=1e-6 \
  --total_training_time=96.0 \
  --early_exit_time_marker=96.0 \
  --output_dir=$check_point_dir/$2 \
  --print_steps=100 \
  --num_epochs_between_checkpoints=100 \
  --job_name=pretraining_experiment \
  --project_name=budget-bert-pretraining \
  --validation_epochs=12 \
  --validation_epochs_begin=1 \
  --validation_epochs_end=1 \
  --validation_begin_proportion=0.05 \
  --validation_end_proportion=0.01 \
  --validation_micro_batch=16 \
  --deepspeed \
  --data_loader_type=dist \
  --do_validation \
  --early_stop_time=180 \
  --early_stop_eval_loss=6 \
  --seed=42 \
  --n_local=5 \
  --mpo_config=${base_dir}/mpo_shape_config.json \
  --runname=$2 $3"
  nohup deepspeed --master_port=29541 $base_dir/run_pretraining.py \
      ${COMMON_ARGS} > $base_dir/logs/$2_$(date "+%Y%m%d-%H%M%S").log 2>&1 &

}
########## 12 layer albert
run_pretrain 0,1 run_mpo_12 --fp16\ --num_hidden_groups=12\ --validation_epochs=12\ --add_nsp\ --optimizer_type=lamb\ --lr=0.00000176\ --lr_schedule=step\ --max_steps=125000\ --vocab_size=30000\ --tokenizer_name=albert-base-v2\ --model_type=albert-mlm-sop\ --warmup_proportion=0.025\ --hidden_dropout_prob=0.0\ --attention_probs_dropout_prob=0.0\ --hidden_size=768\ --intermediate_size=3072\ --num_attention_heads=12\ --num_hidden_layers=12\ --share_layer=0,12\ --n_local=5\ --mpo_layers=FFN_1,FFN_2,attention\ --linear_trunc=100000\ --attention_trunc=100000\ --load_layer=noload\ --train_batch_size=4096\ --train_micro_batch_size_per_gpu=32\ --validation_micro_batch=4\ --num_epochs_between_checkpoints=20\ --dataset_path=${train_data_dir}\ --total_training_time=24.0\ --early_exit_time_marker=24.0\ --load_from_albert=${albert_checkpoint_path}