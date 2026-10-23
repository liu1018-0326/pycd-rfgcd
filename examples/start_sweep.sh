#!/bin/bash
# start_sweep.sh

# 设置参数
PROJECT_NAME="YOUR_WANDB_PROJECT"
MODEL_NAME="icdm"
DATASET="math2" 
FOLDS="1,0"
GPU_IDS="2"
AGENT_SCRIPT="run_agents.sh"
LOG_FILE="wandb_agents.log"
TEMP_FILE="sweep_creation_output.txt"

# 确保加载wandb API key
if [ -z "$WANDB_API_KEY" ]; then
    if [ -f "../configs/wandb.json" ]; then
        WANDB_API_KEY=$(cat ../configs/wandb.json | grep "api_key" | cut -d'"' -f4)
        export WANDB_API_KEY
    else
        echo "Error: WANDB_API_KEY not set and ../configs/wandb.json not found"
        exit 1
    fi
fi

echo "Using WANDB_API_KEY: $WANDB_API_KEY"

# 第一步：生成sweep配置
echo "Step 1: Generating sweep configurations..."
python generate_sweeps.py --model_names $MODEL_NAME --dataset $DATASET --folds $FOLDS --project_name "$PROJECT_NAME"

# 第二步：逐个创建sweep并直接捕获ID
echo "Step 2: Creating sweeps and capturing IDs..."
> sweep_ids.txt  # 创建或清空文件

# 直接设置环境变量，避免在每行命令中重复
export WANDB_API_KEY

# 读取all_start.sh中的每一行，但是忽略WANDB_API_KEY部分
while read -r line; do
    # 提取实际的命令部分（去掉环境变量设置）
    CMD=$(echo "$line" | sed "s/WANDB_API_KEY=[^ ]* //")
    echo "Executing: $CMD"
    
    # 执行命令并同时捕获标准输出和标准错误
    $CMD |& tee $TEMP_FILE
    
    # 从输出中提取sweep ID
    SWEEP_ID=$(grep "wandb agent" $TEMP_FILE | awk '{print $NF}')
    
    if [ -n "$SWEEP_ID" ]; then
        echo "Found sweep ID: $SWEEP_ID"
        echo "$SWEEP_ID" >> sweep_ids.txt
    else
        echo "Warning: Could not extract sweep ID from the output"
        echo "Command output:"
        cat $TEMP_FILE
    fi
done < all_start.sh

# 检查是否获取到了sweep ID
if [ ! -s sweep_ids.txt ]; then
    echo "Error: Failed to extract any sweep IDs. Cannot continue."
    exit 1
fi

echo "Step 3: Generating agent commands with background execution..."
echo "Found the following sweep IDs:"
cat sweep_ids.txt

# 创建运行脚本，所有输出重定向到同一个日志文件
echo "#!/bin/bash" > $AGENT_SCRIPT
echo "" >> $AGENT_SCRIPT
echo "# 清空或创建日志文件" >> $AGENT_SCRIPT
echo "> $LOG_FILE" >> $AGENT_SCRIPT
echo "" >> $AGENT_SCRIPT

# 确保GPU_IDS不为空，如果为空则使用默认值
if [ -z "$GPU_IDS" ]; then
    echo "Warning: GPU_IDS is empty, using default GPU 0"
    GPU_IDS="0"
fi

# 将GPU IDS转换为数组
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
GPU_COUNT=${#GPU_ARRAY[@]}

# 确保GPU_COUNT不为0
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "Error: Failed to parse GPU IDs. Using default GPU 0."
    GPU_ARRAY=("0")
    GPU_COUNT=1
fi

echo "Using ${GPU_COUNT} GPUs: ${GPU_ARRAY[*]}"

# 读取sweep IDs并添加到脚本
COUNT=0
while read -r SWEEP_ID; do
    GPU_IDX=$((COUNT % GPU_COUNT))
    GPU=${GPU_ARRAY[$GPU_IDX]}
    
    # 使用 >> 确保所有输出附加到同一个日志文件
    COMMAND="CUDA_VISIBLE_DEVICES=$GPU nohup wandb agent $SWEEP_ID >> $LOG_FILE 2>&1 &"
    echo "$COMMAND" >> $AGENT_SCRIPT
    echo "Added background command for sweep $SWEEP_ID on GPU $GPU"
    
    COUNT=$((COUNT + 1))
done < sweep_ids.txt

# 添加等待命令以显示所有进程已启动
echo "" >> $AGENT_SCRIPT
echo "echo 'All sweep agents have been started in background.'" >> $AGENT_SCRIPT
echo "echo 'Check $LOG_FILE for output from all agents.'" >> $AGENT_SCRIPT

# 添加执行权限
chmod +x $AGENT_SCRIPT

echo ""
# echo "Agent run script has been generated in '$AGENT_SCRIPT'."
# echo "To launch all agents in background with output redirection to $LOG_FILE, run:"
echo "  bash $AGENT_SCRIPT"
echo ""
echo "The script contains the following commands:"
cat $AGENT_SCRIPT
echo ""
echo "Sweep generation complete!"

# 清理临时文件
rm -f $TEMP_FILE