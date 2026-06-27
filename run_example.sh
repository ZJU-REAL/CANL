PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"



## fixed parameters
TASK_TYPE="points2point2bbox_reward"
is_reward_customized_from_vlm_module=True
points2point_method="confidence"
is_binary_reward=True
scale=False
positive_learning=False
answer_tag="[]"
adaptive_batch=False
normalize='pre'

## flexible parameters
model_size="3B"
per_device_train_batch_size=1
nproc_per_node=4
num_train_epochs=1
num_generations=4
gradient_accumulation_steps=4
learning_rate=1e-6
temperature=1.0
threshold=0.05
downsample_fac=2
negative_learning=True # if CAL, set to False
positive_reward=1
negative_reward=0
save_steps=50

n2=$((per_device_train_batch_size*nproc_per_node))
n1=$((n2*downsample_fac))



if [ $n2 -ne $num_generations ]; then
  echo "per_device_train_batch_size * nproc_per_node must equal to num_generations"
  exit 1
fi

export REPO_HOME="${PROJECT_ROOT}"
export EXP_NAME="example_exp" # TODO: change this to your own experiment name
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"
# export NCCL_P2P_DISABLE=1
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL

echo "REPO_HOME: $REPO_HOME"
data_paths="./dataset/pro/rl/android_studio_macos.jsonl\
:./dataset/pro/rl/autocad_windows.jsonl\
:./dataset/pro/rl/blender_windows.jsonl\
:./dataset/pro/rl/davinci_macos.jsonl\
:./dataset/pro/rl/eviews_windows.jsonl\
:./dataset/pro/rl/excel_macos.jsonl\
:./dataset/pro/rl/fruitloops_windows.jsonl\
:./dataset/pro/rl/illustrator_windows.jsonl\
:./dataset/pro/rl/inventor_windows.jsonl\
:./dataset/pro/rl/linux_common_linux.jsonl\
:./dataset/pro/rl/macos_common_macos.jsonl\
:./dataset/pro/rl/matlab_macos.jsonl\
:./dataset/pro/rl/origin_windows.jsonl\
:./dataset/pro/rl/photoshop_windows.jsonl\
:./dataset/pro/rl/powerpoint_windows.jsonl\
:./dataset/pro/rl/premiere_windows.jsonl\
:./dataset/pro/rl/pycharm_macos.jsonl\
:./dataset/pro/rl/quartus_windows.jsonl\
:./dataset/pro/rl/solidworks_windows.jsonl\
:./dataset/pro/rl/stata_windows.jsonl\
:./dataset/pro/rl/unreal_engine_windows.jsonl\
:./dataset/pro/rl/vivado_windows.jsonl\
:./dataset/pro/rl/vmware_macos.jsonl\
:./dataset/pro/rl/vscode_macos.jsonl\
:./dataset/pro/rl/windows_common_windows.jsonl\
:./dataset/pro/rl/word_macos.jsonl" 

image_folders="your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images\
:your/path/to/ScreenSpot-Pro/images"

model_path="your/model/path/Qwen2.5-VL-${model_size}-Instruct"
echo "image_folders: $image_folders"
cd ${REPO_HOME}/src/open-r1-multimodal

mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log

# export WANDB_DISABLED=true


torchrun --nproc_per_node=$nproc_per_node \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="20002" \
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
    --data_seed 88 \
    --save_steps $save_steps \
    --num_generations $num_generations \
    --max_completion_length 64 \
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
    --save_only_model True
echo "Training completed for ${EXP_NAME}"
