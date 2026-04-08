import pandas as pd
import numpy as np
import glob
import os

# ==========================================
# 1. 设置输入和输出文件夹路径
# ==========================================
input_folder = "E:\\Project\\6CCE3EEP_FYP\\Analysing\\PC\\"

# 创建一个专门存放截断后数据的全新文件夹
output_folder = os.path.join(input_folder, "Adjusted0_Data")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"已创建新文件夹: {output_folder}")

# 获取所有原始 Tracking 文件
file_list = glob.glob(os.path.join(input_folder, "*_TrackingData_*.csv"))
print(f"找到 {len(file_list)} 个原始数据文件，开始处理...")

# ==========================================
# 2. 解码目标的真实绝对坐标 (32宫格)
# ==========================================
azimuth_map = [0, 45, 90, 135, 180, -135, -90, -45]
elevation_map = [-30, 0, 30, 60]

# ==========================================
# 3. 逐个文件处理并保存
# ==========================================
for file in file_list:
    # 读取单个被试的文件
    df = pd.read_csv(file)
    
    # 临时计算出当前文件中所有行的目标真实坐标
    df['Target_Yaw'] = df['TargetNumber'].apply(lambda x: azimuth_map[int(x) // 4])
    df['Target_Pitch'] = df['TargetNumber'].apply(lambda x: elevation_map[int(x) % 4])
    
    truncated_trials = []
    
    # 按 Phase 和 Trial 分组，逐个试次进行截断
    for (phase, trial), group in df.groupby(['Phase', 'Trial']):
        group = group.sort_values('TimeStamp_sec').reset_index(drop=True)
        
        target_yaw = group['Target_Yaw'].iloc[0]
        target_pitch = group['Target_Pitch'].iloc[0]
        
        # 计算每一帧与目标的欧几里得角距离
        yaw_diff = np.abs((group['Yaw_deg'] - target_yaw + 180) % 360 - 180)
        pitch_diff = np.abs(group['Pitch_deg'] - target_pitch)
        angular_distance = np.sqrt(yaw_diff**2 + pitch_diff**2)
        
        # 找到距离最近的那一帧的索引
        closest_idx = angular_distance.idxmin()
        
        # 截断数据！只保留从 0 秒到最接近目标瞬间的数据
        truncated_group = group.loc[:closest_idx].copy()
        truncated_trials.append(truncated_group)
    
    # 将该被试所有截断后的试次重新拼合在一起
    df_truncated = pd.concat(truncated_trials, ignore_index=True)
    
    # 删除临时添加的辅助列，保持与原实验数据结构完全一致
    df_truncated = df_truncated.drop(columns=['Target_Yaw', 'Target_Pitch'])
    
    # ==========================================
    # 4. 生成新文件名并保存到新文件夹
    # ==========================================
    # 提取原文件名 (例如 P001_TrackingData_20260402_131014.csv)
    original_filename = os.path.basename(file)
    # 在文件名末尾加上 _Truncated 以示区别
    new_filename = original_filename.replace(".csv", "_Truncated.csv")
    new_filepath = os.path.join(output_folder, new_filename)
    
    # 保存为 CSV 文件
    df_truncated.to_csv(new_filepath, index=False)
    print(f"已处理并保存: {new_filename}")

print(f"\n🎉 完美！所有截断后的数据均已保存在:\n{output_folder}")