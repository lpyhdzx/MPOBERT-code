set -x
base_dir=/home/liupeiyu/MPO-albert_cuda
check_point_dir=/mnt/liupeiyu/checkpoint/academic_bert_154
data_dir_base=/mnt/liupeiyu/nlp_data/GLUE

echo $gpu_num
function run_task() {
  export CUDA_VISIBLE_DEVICES=$1
  nohup python $base_dir/pbt_albert.py --runname=$2 $3 > log_pbt/$2_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}
run_task 6 rte_param_bert --task=rte\ --model_name=/mnt/liupeiyu/checkpoint/academic-budget-bert/run_mpo_12_albertmini_initCT_v2_lr00001_4/pretraining_experiment-/epoch1000000_step10324 # bert large