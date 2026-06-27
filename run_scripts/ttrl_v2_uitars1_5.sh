PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# 实验名称
model_size="7B"
benchmark="v2"


# 固定不动
TASK_TYPE="points2point2bbox_reward"
is_reward_customized_from_vlm_module=True
scale=False
points2point_method="confidence"
is_binary_reward=True
positive_learning=False
answer_tag="()"
# 核心参数
per_device_train_batch_size=4
nproc_per_node=2
num_train_epochs=1
num_generations=8
gradient_accumulation_steps=8
learning_rate=1e-6
temperature=1.0
threshold=0.05
downsample_fac=2
negative_learning=True
positive_reward=1
negative_reward=0
normalize='pre'
adaptive_batch=False
# 其他
save_steps=50

n2=$((per_device_train_batch_size*nproc_per_node))
n1=$((n2*downsample_fac))
# 如果用了negative_learning，自动将positive_reward置为0，且不归一化
[ "$negative_learning" = "True" ] && learning_method="-nl" || learning_method=""

if [ "$normalize" = "pre" ]; then
  adv_method="-prenorm"
elif [ "$normalize" = "later" ]; then
  adv_method="-norm"
else
  adv_method=""
fi
if [ $n2 -ne $num_generations ]; then
  echo "per_device_train_batch_size * nproc_per_node must equal to num_generations"
  exit 1
fi
[ "$adaptive_batch" = "True" ] && adaptive_batch_method="-adpb" || adaptive_batch_method=""
export REPO_HOME="${PROJECT_ROOT}"
export EXP_NAME="UITARS1_5-${model_size}${learning_method}-${positive_reward}_${negative_reward}-${n1}_${n2}-bs${gradient_accumulation_steps}-${threshold}-${temperature}${adv_method}${adaptive_batch_method}" # TODO: change this to your own experiment name
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
export NCCL_P2P_DISABLE=1
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL

echo "REPO_HOME: $REPO_HOME"
data_paths="/home/liuyizhou/benchmark/ScreenSpot-v2/mobile_rl.jsonl:/home/liuyizhou/benchmark/ScreenSpot-v2/desktop_rl.jsonl:/home/liuyizhou/benchmark/ScreenSpot-v2/web_rl.jsonl" 
image_folders="/home/liuyizhou/benchmark/ScreenSpot-v2/screenspotv2_image:/home/liuyizhou/benchmark/ScreenSpot-v2/screenspotv2_image:/home/liuyizhou/benchmark/ScreenSpot-v2/screenspotv2_image"
model_path="/home/real/GUI_Agent_Infra/models/UI-TARS-1.5-7B"
echo "image_folders: $image_folders"
cd ${REPO_HOME}/src/open-r1-multimodal

mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log

# export WANDB_DISABLED=true


torchrun --nproc_per_node=$nproc_per_node \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="20001" \
  src/open_r1/grpo_jsonl.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/checkpoints/rl/${benchmark}/${EXP_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs $num_train_epochs \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --data_seed 42 \
    --save_steps $save_steps \
    --num_generations $num_generations \
    --max_completion_length 128 \
    --beta 0.04 \
    --report_to tensorboard \
    --dataset-name this_is_not_used \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3_offload.json \
    --learning_rate $learning_rate \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --reward_funcs accuracy \
    --top_k 50 \
    --top_p 1 \
    --temperature $temperature \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --scale $scale \
    --points2point_method $points2point_method \
    --is_binary_reward $is_binary_reward \
    --threshold $threshold \
    --downsample_fac $downsample_fac \
    --negative_learning $negative_learning \
    --positive_learning $positive_learning \
    --positive_reward $positive_reward \
    --negative_reward $negative_reward \
    --normalize $normalize \
    --adaptive_batch $adaptive_batch \
    --answer_tag $answer_tag \
    --model_size $model_size \
    --model_type "ui-tars-1.5"
echo "Training completed for ${EXP_NAME}"
