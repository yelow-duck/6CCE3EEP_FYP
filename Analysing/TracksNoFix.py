import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os

# ==========================================
# 0. 基础设置与中文字体支持
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 读取数据并创建专门的保存文件夹
# ==========================================
folder_path = "E:\\Project\\6CCE3EEP_FYP\\Analysing\\PC\\"

# 自动抓取 Tracking 数据
file_list = glob.glob(os.path.join(folder_path, "*_TrackingData_*.csv"))
df_all = pd.concat([pd.read_csv(f) for f in file_list], ignore_index=True)

# 【核心】：在 PC 文件夹下创建一个专门存放轨迹图的新文件夹
output_dir = os.path.join(folder_path, "Trajectories_Plots")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建轨迹图保存文件夹: {output_dir}")

# ==========================================
# 2. 解码目标的真实绝对坐标 (用于在图上画出目标五角星)
# ==========================================
azimuth_map = [0, 45, 90, 135, 180, -135, -90, -45]
elevation_map = [-30, 0, 30, 60]
df_all['Target_Yaw'] = df_all['TargetNumber'].apply(lambda x: azimuth_map[int(x) // 4])
df_all['Target_Pitch'] = df_all['TargetNumber'].apply(lambda x: elevation_map[int(x) % 4])

# ==========================================
# 3. 分组并批量绘图
# ==========================================
# 按照 被试、阶段、试次 进行分组
groups = df_all.groupby(['ParticipantID', 'Phase', 'Trial'])

print(f"即将生成 {len(groups)} 张轨迹图，请稍候...")

count = 0
for (pid, phase, trial), group_data in groups:
    # 确保时间是按顺序排列的，避免轨迹线乱穿
    group_data = group_data.sort_values('TimeStamp_sec')
    
    # 【文件命名逻辑处理】
    # 将 'HRTF_A' 变成 'A', 'HRTF_B' 变成 'B'
    phase_letter = phase.split('_')[-1] 
    # 试次往往从 0 开始，加 1 后补齐两位数，例如 0 变成 '01', 9 变成 '10'
    trial_str = f"{int(trial) + 1:02d}" 
    
    # 拼接最终文件名：例如 P001A01.png
    filename = f"{pid}{phase_letter}{trial_str}.png"
    filepath = os.path.join(output_dir, filename)
    
    # ================= 绘图核心部分 =================
    # 使用稍小的尺寸加速生成并节省硬盘空间
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 1. 画出连续的运动轨迹连线 (灰色偏黑，稍微透明一点)
    ax.plot(group_data['Yaw_deg'], group_data['Pitch_deg'], color='#2C3E50', alpha=0.6, linewidth=1)
    
    # 2. 画出起点 (绿色)
    start_yaw, start_pitch = group_data['Yaw_deg'].iloc[0], group_data['Pitch_deg'].iloc[0]
    ax.scatter(start_yaw, start_pitch, color='#2ECC71', s=80, label='Start (0s)', zorder=5)
    
    # 3. 画出终点 (红色)
    end_yaw, end_pitch = group_data['Yaw_deg'].iloc[-1], group_data['Pitch_deg'].iloc[-1]
    ax.scatter(end_yaw, end_pitch, color='#E74C3C', s=80, label='End (Located)', zorder=5)
    
    # 4. 画出系统的真实目标位置 (金色五角星)
    target_yaw = group_data['Target_Yaw'].iloc[0]
    target_pitch = group_data['Target_Pitch'].iloc[0]
    ax.scatter(target_yaw, target_pitch, color='#F1C40F', marker='*', s=250, edgecolors='black', label='Target Position', zorder=6)
    
    # ================= 美化与保存 =================
    ax.set_title(f"Head Trajectory: {pid} | {phase} | Trial {trial_str}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Horizontal Yaw (deg)")
    ax.set_ylabel("Vertical Pitch (deg)")
    
    # 固定坐标轴的范围，保证所有的图比例尺一样，方便直接对比！
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    
    # 强制固定刻度
    ax.set_xticks([-135, -90, -45, 0, 45, 90, 135, 180])
    ax.set_yticks([-60, -30, 0, 30, 60])
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    # dpi=150 能保证足够清晰的同时不会让几百张图片占用几个G的空间
    plt.savefig(filepath, dpi=150)
    
    # 【防内存崩溃核心】：画完必须关掉这张图表
    plt.close(fig)
    
    count += 1
    # 每生成 50 张打印一次进度，让你知道程序没卡死
    if count % 50 == 0:
        print(f"已生成 {count} 张图片...")

print(f"\n🎉 恭喜！全部 {count} 张轨迹图已成功生成并保存在：\n{output_dir}")