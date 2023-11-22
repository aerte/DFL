#!/bin/bash
# trap "exit" INT
trap 'kill $(jobs -p)' EXIT
# # SBATCH --job-name=per_sample
# #SBATCH --output=cifar_per_sample-%J.out
# #SBATCH --cpus-per-task=2
# #SBATCH --time=12:00:00
# #SBATCH --mem=42gb
# #SBATCH --nodes=1
# #SBATCH --ntasks=10
# #SBATCH --gres=gpu:3
# #SBATCH --mail-user=blia@dtu.dk
# #SBATCH --mail-type=END,FAIL
# #SBATCH --export=ALL
# FD
## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

# https://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@github.com/aerte/DFL.git

lr_group="0.1"
n_clients=10
split=iid
local_epoch=20
method=check_zeta
non_iid_alpha=0.1
dataset=cifar10
model_type=m_cnn
version=6
num_rounds=30
sigma=0
start_round=0
start_client=0
end_client=9

# num2=1
# num3=6
gpu_index = 1

# echo ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}

for s_lr in $lr_group
do
    python mnist_utils.py --n_clients "$n_clients" --split "$split" --sigma "$sigma" --num_local_epochs "$local_epoch" \
            --method "$method" --version "$version" --lr "$s_lr" \
            --num_rounds "$num_rounds" --dataset "$dataset" --model_type "$model_type" --non_iid_alpha "$non_iid_alpha" --start_round "$start_round"

    for round in $(seq "$start_round" 1 "$num_rounds")
    do
        for i in $(seq "$start_client" 1 "$end_client")
        do
            #if [ "$i" -lt "$num2" ]; then
            #    gpu_index=1
            #elif [ "$i" -ge "$num2" ] && [ "$i" -lt "$num3" ]; then
            #    gpu_index=3
            #fi
            echo "|GPU INDEX|CLIENT INDEX|${gpu_index}|${i}"
            export CUDA_VISIBLE_DEVICES="$gpu_index"
            python train_cifar10_efficient.py --n_clients "$n_clients" --split "$split" --sigma "$sigma" --num_local_epochs "$local_epoch" \
                --method "$method" --version "$version" --lr "$s_lr" \
                --num_rounds "$num_rounds" --use_local_id "$i" --dataset "$dataset" --opt client \
                --model_type "$model_type" --non_iid_alpha "$non_iid_alpha" --start_round "$start_round" --round "$round" &
        done
        wait
    done

done