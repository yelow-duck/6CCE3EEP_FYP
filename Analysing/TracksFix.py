import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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

# 创建存放【渐变色轨迹图】的新文件夹
output_dir = os.path.join(folder_path, "Trajectories_TimeHeatmap")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建渐变色轨迹图保存文件夹: {output_dir}")

# ==========================================
# 2. 解码目标的真实绝对坐标
# ==========================================
azimuth_map = [0, 45, 90, 135, 180, -135, -90, -45]
elevation_map = [-30, 0, 30, 60]
df_all['Target_Yaw'] = df_all['TargetNumber'].apply(lambda x: azimuth_map[int(x) // 4])
df_all['Target_Pitch'] = df_all['TargetNumber'].apply(lambda x: elevation_map[int(x) % 4])

# ==========================================
# 3. 分组并批量绘图 (带时间渐变与智能截断)
# ==========================================
groups = df_all.groupby(['ParticipantID', 'Phase', 'Trial'])
print(f"即将生成 {len(groups)} 张渐变色轨迹图，请稍候...")

count = 0
for (pid, phase, trial), group_data in groups:
    # 确保时间戳排序正确
    group_data = group_data.sort_values('TimeStamp_sec').reset_index(drop=True)
    
    target_yaw = group_data['Target_Yaw'].iloc[0]
    target_pitch = group_data['Target_Pitch'].iloc[0]
    target_number = group_data['TargetNumber'].iloc[0] # 获取目标编号
    
    # -----------------------------------------------------
    # 【1. 计算距离并截断轨迹】
    yaw_diff = np.abs((group_data['Yaw_deg'] - target_yaw + 180) % 360 - 180)
    pitch_diff = np.abs(group_data['Pitch_deg'] - target_pitch)
    angular_distance = np.sqrt(yaw_diff**2 + pitch_diff**2)
    closest_idx = angular_distance.idxmin()
    group_data = group_data.loc[:closest_idx]
    # -----------------------------------------------------
    
    # 【2. 命名标准更新：加入 TargetNumber 并补零】
    phase_letter = phase.split('_')[-1] 
    trial_str = f"{int(trial) + 1:02d}"
    target_str = f"{int(target_number):02d}" # 不足两位数用 0 补全
    
    # 最终文件名例如：P001A01_T25.png
    filename = f"{pid}{phase_letter}{trial_str}_T{target_str}.png"
    filepath = os.path.join(output_dir, filename)
    
    # ================= 绘图核心部分 =================
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # 提取 X (Yaw), Y (Pitch) 和 T (Time) 的数组
    x = group_data['Yaw_deg'].values
    y = group_data['Pitch_deg'].values
    t = group_data['TimeStamp_sec'].values
    
    # 【3. 绘制随时间渐变的热力图轨迹线】
    # 将离散的点组合成线段 (Segments)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # 创建 LineCollection 并映射颜色 (viridis 是经典的蓝-绿-黄渐变色)
    norm = plt.Normalize(t.min(), t.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(t)
    lc.set_linewidth(2.5)
    lc.set_alpha(0.8)
    line = ax.add_collection(lc)
    
    # 添加一个颜色条 (Colorbar) 解释时间流逝
    cbar = fig.colorbar(line, ax=ax, pad=0.02)
    cbar.set_label('Elapsed Time (seconds)', rotation=270, labelpad=15)
    
    # 画起点和截断终点 (为了不和轨迹混淆，改用黑白描边的样式)
    start_yaw, start_pitch = x[0], y[0]
    ax.scatter(start_yaw, start_pitch, color='white', edgecolors='black', s=60, label='Start', zorder=5)
    
    end_yaw, end_pitch = x[-1], y[-1]
    ax.scatter(end_yaw, end_pitch, color='black', edgecolors='white', s=60, label='End (Truncated)', zorder=5)
    
    # 画真实目标位置
    ax.scatter(target_yaw, target_pitch, color='#E74C3C', marker='*', s=250, edgecolors='black', label='Target', zorder=6)
    
    # ================= 美化与保存 =================
    ax.set_title(f"Time-Encoded Trajectory: {pid} | {phase} | Trial {trial_str} | Target {target_str}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Horizontal Yaw (deg)")
    ax.set_ylabel("Vertical Pitch (deg)")
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks([-135, -90, -45, 0, 45, 90, 135, 180])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig) # 释放内存
    
    count += 1
    if count % 50 == 0:
        print(f"已生成 {count} 张带渐变色的轨迹图...")

print(f"\n🎉 全部 {count} 张图已完成！请前往文件夹查看：\n{output_dir}")