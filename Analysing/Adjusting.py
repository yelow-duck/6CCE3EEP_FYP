import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import glob
import os
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 0. 基础设置与中文字体支持
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 读取【截断后】的纯净实验数据
# ==========================================
folder_path = "E:\\Project\\6CCE3EEP_FYP\\Analysing\\PC\\Adjusted0_Data\\"
file_list = glob.glob(os.path.join(folder_path, "*_TrackingData_*.csv"))

df_all = pd.concat([pd.read_csv(f) for f in file_list], ignore_index=True)

# 创建存放【修复Bug后的渐变色轨迹图】的新文件夹
output_dir = os.path.join(folder_path, "Trajectories_Heatmap_Fixeda")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 2. 解码目标的真实绝对坐标
# ==========================================
azimuth_map = [0, 45, 90, 135, 180, -135, -90, -45]
elevation_map = [-30, 0, 30, 60]

# ==========================================
# 3. 分组并批量绘图 
# ==========================================
groups = df_all.groupby(['ParticipantID', 'Phase', 'Trial'])
print(f"即将生成 {len(groups)} 张轨迹图，请稍候...")

count = 0
for (pid, phase, trial), group_data in groups:
    group_data = group_data.sort_values('TimeStamp_sec').reset_index(drop=True)
    
    target_number = group_data['TargetNumber'].iloc[0]
    target_yaw = azimuth_map[int(target_number) // 4]
    target_pitch = elevation_map[int(target_number) % 4]
    
    # 提取 X, Y, T 的数组并确保格式为 float
    x = group_data['Yaw_deg'].values.astype(float)
    y = group_data['Pitch_deg'].values.astype(float)
    t = group_data['TimeStamp_sec'].values.astype(float)
    
    # -----------------------------------------------------
    # 【修复 Bug 核心逻辑：自动断开跨边界的连线】
    # 1. 计算相邻两帧之间的 Yaw 角度差
    diffs = np.abs(np.diff(x))
    
    # 2. 如果角度差大于 180 度 (甚至更严格一点设为 100度)，说明发生了左右边界的穿越
    jumps = diffs > 100 
    
    # 3. 获取发生穿越的索引位置
    jump_indices = np.where(jumps)[0] + 1
    
    # 4. 在穿越点强行插入 NaN，迫使 Matplotlib 断开连线
    x = np.insert(x, jump_indices, np.nan)
    y = np.insert(y, jump_indices, np.nan)
    t = np.insert(t, jump_indices, np.nan)
    # -----------------------------------------------------
    
    # 文件命名逻辑
    phase_letter = phase.split('_')[-1] 
    trial_str = f"{int(trial) + 1:02d}"
    target_str = f"{int(target_number):02d}" 
    filename = f"{pid}{phase_letter}{trial_str}_T{target_str}.png"
    filepath = os.path.join(output_dir, filename)
    
    # ================= 绘图核心部分 =================
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # 绘制随时间渐变的热力图轨迹线
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm = plt.Normalize(np.nanmin(t), np.nanmax(t))
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(t)
    lc.set_linewidth(2.5)
    lc.set_alpha(0.8)
    line = ax.add_collection(lc)
    
    # 颜色条
    cbar = fig.colorbar(line, ax=ax, pad=0.02)
    cbar.set_label('Elapsed Time (seconds)', rotation=270, labelpad=15)
    
    # 画起点和终点
    start_yaw, start_pitch = group_data['Yaw_deg'].iloc[0], group_data['Pitch_deg'].iloc[0]
    ax.scatter(start_yaw, start_pitch, color='white', edgecolors='black', s=60, label='Start', zorder=5)
    
    end_yaw, end_pitch = group_data['Yaw_deg'].iloc[-1], group_data['Pitch_deg'].iloc[-1]
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
    plt.close(fig) 
    
    count += 1
    if count % 50 == 0:
        print(f"已生成 {count} 张修复后的轨迹图...")

print(f"\n🎉 全部 {count} 张图已完成！没有横线的纯净版已保存在：\n{output_dir}")