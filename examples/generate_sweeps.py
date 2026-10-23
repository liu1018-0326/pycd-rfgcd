#!/usr/bin/env python
import os
import json
import argparse
import yaml

def str2bool(s):
    return s.lower() == "true"

def main(params):
    src_dir = params["src_dir"]
    project_name = params["project_name"]
    dataset = params["dataset"]
    model_names = params["model_names"]
    folds = params["folds"]
    all_dir = params["all_dir"]
    launch_file = params["launch_file"]
    
    # 确保输出目录存在
    if not os.path.exists(all_dir):
        os.makedirs(all_dir)
    
    # 加载wandb配置
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    if WANDB_API_KEY is None:
        try:
            with open("../configs/wandb.json") as fin:
                wandb_config = json.load(fin)
                WANDB_API_KEY = wandb_config["api_key"]
        except Exception as e:
            print(f"Error loading wandb config: {e}")
            WANDB_API_KEY = "YOUR_API_KEY"
    
    print(f"Using WANDB_API_KEY: {WANDB_API_KEY}")
    
    with open(launch_file, "w") as fallsh:
        pre = f"WANDB_API_KEY={WANDB_API_KEY} wandb sweep "
        
        # 遍历所有数据集
        for dataset_name in dataset.split(","):
            # 遍历所有模型
            for model in model_names.split(","):
                # 构建源yaml文件路径
                fpath = os.path.join(src_dir, f"{model}.yaml")
                
                # 检查源文件是否存在
                if not os.path.exists(fpath):
                    print(f"Warning: Source file {fpath} does not exist. Skipping.")
                    continue
                
                # 读取源YAML文件内容
                with open(fpath, "r") as fin:
                    yaml_content = fin.read()
                
                # 解析YAML为Python对象
                try:
                    yaml_dict = yaml.safe_load(yaml_content)
                except Exception as e:
                    print(f"Error parsing YAML in {fpath}: {e}")
                    continue
                
                # 遍历所有fold
                for fold in folds.split(","):
                    # 创建目标yaml文件名
                    fname = f"{dataset_name}_{model}_{fold}.yaml"
                    ftarget = os.path.join(all_dir, fname)
                    
                    # 修改YAML对象
                    modified_yaml = yaml_dict.copy()
                    
                    # 检查参数中是否有dataset_name或dataset并更新
                    if 'parameters' in modified_yaml:
                        if 'dataset_name' in modified_yaml['parameters']:
                            # 如果有dataset_name，更新它
                            modified_yaml['parameters']['dataset_name']['values'] = [dataset_name]
                            print(f"Updated dataset_name to {dataset_name}")
                        elif 'dataset' in modified_yaml['parameters']:
                            # 如果有dataset，更新它
                            modified_yaml['parameters']['dataset']['values'] = [dataset_name]
                            print(f"Updated dataset to {dataset_name}")
                        else:
                            # 如果两者都没有，输出警告
                            print(f"Warning: Neither dataset_name nor dataset found in {fpath}")
                            # 添加dataset_name参数
                            modified_yaml['parameters']['dataset_name'] = {'values': [dataset_name]}
                            print(f"Added dataset_name parameter with value {dataset_name}")
                    
                    # 更新fold值
                    if 'parameters' in modified_yaml and 'fold' in modified_yaml['parameters']:
                        modified_yaml['parameters']['fold']['values'] = [int(fold)]
                    
                    # 添加sweep名称
                    modified_yaml_with_name = {
                        'name': f"{dataset_name}_{model}_{fold}"
                    }
                    modified_yaml_with_name.update(modified_yaml)
                    
                    # 写入目标文件
                    with open(ftarget, "w") as fout:
                        yaml.dump(modified_yaml_with_name, fout, default_flow_style=False)
                    
                    print(f"Created sweep config: {ftarget}")
                    
                    # 添加sweep命令到启动脚本
                    fallsh.write(f"{pre} {ftarget}")
                    if project_name:
                        fallsh.write(f" --project {project_name}")
                    fallsh.write("\n")
        
        # 添加执行权限
        os.chmod(launch_file, 0o755)
        print(f"Generated sweep launch script: {launch_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default="./seedwandb/")
    parser.add_argument("--project_name", type=str, default="pycd-experiments")
    parser.add_argument("--dataset", type=str, default="assist2009")
    parser.add_argument("--model_names", type=str, default="neuralcdm")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--all_dir", type=str, default="all_sweeps")
    parser.add_argument("--launch_file", type=str, default="all_start.sh")
    
    args = parser.parse_args()
    params = vars(args)
    print(params)
    main(params)