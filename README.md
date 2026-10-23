# 环境配置
conda 环境
```
conda create --name=env_name python=3.8.17
conda activate env_name
```

项目环境
```
git clone https://github.com/lycyhrc/PYCD.git
cd xxx/PYCD
pip install -r requirements.txt
pip install -e . # 可选：开发模式安装项目
```


# 数据集预处理
数据集下载链接🔗: https://drive.google.com/drive/folders/1DvJBvYTXmnjl6IeavQ0at1S5FSmTXfDL?usp=sharing

```
cd examples

python data_preprocess.py --dataset_name=assist2009 --min_seq_len=15 --split_mode=1 --time_info=0
```
参数说明:

**题目多次练习策略**
>  现有CD模型对于学生对多个问题的尝试只保留第1次交互记录，这里我们考虑三种情况:

- split_mode=1（默认）: 题目多次交互，保留第一次记录
- split_mode=2: 题目多次交互，计算平均准确率作为标签
- split_mode=3: 题目多次交互，计算平均准确率+题目的全局平均准确率加权平均值作为标签

**时间策略**
>  现有CD模型对于学生的答题序列不考虑时间信息，我们对学生交互序列引入time_info参数按照周划分新学生，如果两个交互时间间隔超过time_info，那么切分为新学生:

- time_info=0（默认）: 不考虑时间信息
- time_info=1: 按交互的时间间隔1周切分
- time_info=2: 按交互的时间间隔2周切分
- time_info=4: 按交互的时间间隔4周切分

*注意：在两种模式下，（切分）数据长度小于min_seq_len=15的学生交互记录都会被删除掉*


# 模型训练与评估

```
python example_neuralcdm.py --dataset assist2009
```

训练完毕，模型存储在`xx/PYCD/examples/model_save`下

# Wandb 调参

本项目支持使用Weights & Biases (wandb)进行超参数搜索和实验跟踪。以下是使用wandb进行模型调优的步骤。

## 前置准备

1. 安装wandb & 注册wandb账号并获取API密钥
   ```bash
   pip install wandb
   ```

2. 配置API密钥
   - 创建配置文件 `configs/wandb.json`
   ```json
   {
      "uid": "用户名"
     "api_key": "wandb API密钥"
   }
   ```

## Sweep配置文件

在`seedwandb`目录中创建模型的yaml配置文件，例如`neuralcdm.yaml`：


此配置文件探索不同的超参数组合，包括隐藏层维度、dropout率、学习率等。

## 自动化调参流程

我们提供了一个自动化脚本`start_sweep.sh`来简化wandb调参流程：

```bash
#!/bin/bash
# 设置参数
PROJECT_NAME="pycd-neuralcdm"  # wandb项目名称
MODEL_NAME="neuralcdm"         # 模型名称
DATASET_NAME="assist2009"      # 数据集名称
FOLDS="0,1,2,3,4"              # 交叉验证折数
GPU_IDS="0,1,2,3"              # 使用的GPU ID
AGENT_SCRIPT="run_agents.sh"   # 生成的agent执行脚本
LOG_FILE="wandb_agents.log"    # 日志文件

# 执行脚本启动sweep
bash start_sweep.sh
```

执行这个脚本将：
1. 生成折特定的sweep配置
2. 创建wandb sweep
3. 生成一个`run_agents.sh`脚本用于启动agent

## 启动Sweep Agent

完成sweep创建后，可以执行生成的`run_agents.sh`脚本来启动agent：

```bash
bash run_agents.sh
```

这将在后台启动所有sweep agent，并将输出重定向到`wandb_agents.log`文件。每个agent将在指定的GPU上运行，自动执行网格搜索中定义的实验。

## 查看实验结果

所有实验结果都会自动上传到wandb平台，你可以在wandb网站上：
- 比较不同超参数组合的性能
- 可视化训练过程
- 分析参数重要性
- 选择最佳模型配置

## 自定义配置

你可以根据需要修改以下参数：
- 模型名称和数据集
- 超参数搜索空间
- GPU分配

例如，要为KaNCD模型创建sweep，只需修改`MODEL_NAME`和对应的yaml配置文件即可。



# 项目结构

```bash
PYCD/
├── examples/                               # 示例和实验脚本
│   ├── data_preprocess.py                  # 数据集处理入口
│   ├── example_dina.py                     # DINA模型示例
│   ├── example_disengcd.py                 # DisengCD模型示例
│   ├── example_hypercdm.py                 # HyperCDM模型示例
│   ├── example_icdm.py                     # ICDM模型示例
│   ├── example_irt.py                      # IRT模型示例
│   ├── example_kancd.py                    # KANCD模型示例
│   ├── example_kscd.py                     # KSCD模型示例
│   ├── example_mirt.py                     # MIRT模型示例
│   ├── example_neuralcdm.py                # NeuralCDM模型示例
│   ├── example_orcdf.py                    # ORCDF模型示例
│   ├── example_rcd.py                      # RCD模型示例
│   ├── example_scd.py                      # SCD模型示例
│   ├── generate_sweeps.py                  # 生成超参数搜索配置
│   ├── run_sweeps.py                       # 运行超参数搜索
│   ├── start_sweep.sh                      # 启动搜索脚本
│   ├── wandb_train_test.py                 # WandB训练测试
│   ├── wandb_utils.py                      # WandB工具函数
│   └── seedwandb/                          # WandB配置文件
│       ├── dina.yaml
│       ├── disengcd.yaml
│       ├── hypercdm.yaml
│       ├── icdm.yaml
│       ├── irt.yaml
│       ├── kancd.yaml
│       ├── kscd.yaml
│       ├── mirt.yaml
│       ├── neuralcdm.yaml
│       ├── orcdf.yaml
│       ├── rcd.yaml
│       └── scd.yaml
├── data/                                   # 数据处理模块
│   ├── __init__.py
│   ├── dataset.py                          # 数据加载器
│   ├── graph_utils.py                      # RCD中建图相关方法
│   ├── utils.py                            # 数据处理工具函数
│   └── assist2009/                         # ASSISTments 2009数据集
│       ├── data.txt
│       ├── id_mapping.json
│       ├── q_matrix.csv
│       ├── que_kc_lookup.txt
│       ├── skill_builder_data_corrected_collapsed.csv
│       ├── test.csv
│       ├── test_problems.json
│       └── train_valid.csv
├── pycd/                                   # 核心包
│   ├── __init__.py
│   ├── configs/                            # 配置模块
│   │   └── __init__.py
│   ├── models/                             # 模型实现
│   │   ├── __init__.py
│   │   ├── base.py                         # 基础模型类
│   │   ├── dina.py                         # DINA模型
│   │   ├── disengcd.py                     # DisengCD模型
│   │   ├── hypercdm.py                     # HyperCDM模型
│   │   ├── icdm.py                         # ICDM模型
│   │   ├── init_model.py                   # 模型初始化
│   │   ├── irt.py                          # IRT模型
│   │   ├── kancd.py                        # KANCD模型
│   │   ├── kscd.py                         # KSCD模型
│   │   ├── mirt.py                         # MIRT模型
│   │   ├── neuralcdm.py                    # NeuralCDM模型
│   │   ├── orcdf.py                        # ORCDF模型
│   │   ├── rcd.py                          # RCD模型
│   │   └── scd.py                          # SCD模型
│   ├── train/                              # 训练模块
│   │   ├── __init__.py
│   │   └── trainer.py                      # 训练器
│   ├── evaluate/                           # 评估模块
│   │   ├── __init__.py
│   │   └── metrics.py                      # 评估指标
│   ├── utils/                              # 工具模块
│   │   ├── __init__.py
│   │   ├── config.py                       # 配置工具
│   │   ├── logging.py                      # 日志工具
│   │   └── utils.py                        # 通用工具函数
│   └── preprocess/                         # 数据预处理模块
│       ├── __init__.py
│       ├── assist2009_preprocess.py        # ASSISTments 2009预处理
│       ├── assist2012_preprocess.py        # ASSISTments 2012预处理
│       ├── assist2017_preprocess.py        # ASSISTments 2017预处理
│       ├── data_proprocess.py              # 通用数据预处理
│       ├── ednet_preprocess.py             # EdNet预处理
│       ├── frcsub_preprocess.py            # FrcSub预处理
│       ├── jiuzhang_g4_g5_g7_preprocess.py # 九章算术预处理
│       ├── jiuzhang_preprocess.py          # 九章算术预处理
│       ├── junyi_preprocess.py             # Junyi预处理
│       ├── math1_preprocess.py             # Math1预处理
│       ├── math2_preprocess.py             # Math2预处理
│       ├── nipd_task34_preprocess.py       # NIPD Task34预处理
│       ├── nips_task34_preprocess.py       # NIPS Task34预处理
│       ├── peiyou_preprocess.py            # 培优预处理
│       ├── slp_math_preprocess.py          # SLP Math预处理
│       ├── split_datasets.py               # 数据集分割
│       └── utils.py                        # 预处理工具函数
├── configs/                                # 全局配置
│   ├── __init__.py
│   └── data_config.json                    # 数据配置
├── setup.py                                # 安装配置
├── requirements.txt                        # 依赖包列表
├── README.md                               # 项目说明
└── .gitignore                              # Git忽略文件

```