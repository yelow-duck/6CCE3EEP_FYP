import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# ==========================================
# 0. 基础设置与中文字体支持
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 读取数据与坐标系解码
# ==========================================
file_path = "你的被试数据_TrackingData_xxxx.csv" # ⚠️ 记得替换为你的真实 CSV 文件路径
df = pd.read_csv(file_path)

azimuth_map = [0, 45, 90, 135, 180, -135, -90, -45]
elevation_map = [-30, 0, 30, 60]
df['Target_Yaw'] = df['TargetNumber'].apply(lambda x: azimuth_map[int(x) // 4])
df['Target_Pitch'] = df['TargetNumber'].apply(lambda x: elevation_map[int(x) % 4])

# ==========================================
# 2. 【核心升级】程序自动检索 Trial 编号
# ==========================================
# 🎯 在这里设定你想观察的“目标编号 (0-31)”
TARGET_TO_OBSERVE = 9 

# 让程序自动去茫茫数据海中寻找，哪些 Trial 测试了这个目标
trials_A = df[(df['Phase'] == 'HRTF_A') & (df['TargetNumber'] == TARGET_TO_OBSERVE)]['Trial'].unique()
trials_B = df[(df['Phase'] == 'HRTF_B') & (df['TargetNumber'] == TARGET_TO_OBSERVE)]['Trial'].unique()

# 安全拦截：防止你输入了一个这名被试没测过的球
if len(trials_A) == 0 or len(trials_B) == 0:
    print(f"❌ 检索失败：被试在 HRTF_A 或 HRTF_B 中没有测试过目标 {TARGET_TO_OBSERVE}！")
    print(f"HRTF_A 包含的试次: {trials_A}")
    print(f"HRTF_B 包含的试次: {trials_B}")
    exit() # 找不到就直接停止程序运行

# 自动提取两组中第一次测到该目标的 Trial 编号
trial_A_num = trials_A[0]
trial_B_num = trials_B[0]

# 根据提取出的编号，切出这两局的连续数据
df_A_single = df[(df['Phase'] == 'HRTF_A') & (df['Trial'] == trial_A_num)].copy()
df_B_single = df[(df['Phase'] == 'HRTF_B') & (df['Trial'] == trial_B_num)].copy()

time_A = df_A_single['TimeStamp_sec'].max()
time_B = df_B_single['TimeStamp_sec'].max()
target_y = df_A_single['Target_Yaw'].iloc[0]
target_p = df_A_single['Target_Pitch'].iloc[0]

print(f"✅ 检索成功！目标 {TARGET_TO_OBSERVE} (Yaw: {target_y}°, Pitch: {target_p}°) 对应：")
print(f"  👉 HRTF A 的第 {trial_A_num} 局 (耗时 {time_A:.2f} 秒)")
print(f"  👉 HRTF B 的第 {trial_B_num} 局 (耗时 {time_B:.2f} 秒)")
print("正在生成轨迹图...")

# ==========================================
# 3. 开始绘图
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
datasets = [df_A_single, df_B_single]
titles = [f"HRTF A: 寻找轨迹 (耗时 {time_A:.2f} 秒)", f"HRTF B: 寻找轨迹 (耗时 {time_B:.2f} 秒)"]

for i in range(2):
    ax = axes[i]
    data = datasets[i]
    x, y, t = data['Yaw_deg'], data['Pitch_deg'], data['TimeStamp_sec']

    # 画灰色轨迹底线
    ax.plot(x, y, color='gray', alpha=0.3, linewidth=1.5, zorder=1)
    # 画时间渐变散点
    sc = ax.scatter(x, y, c=t, cmap='viridis', s=25, alpha=0.9, zorder=2)
    # 标起点与终点
    ax.scatter([0], [0], color='black', marker='X', s=150, label='注视起点 (0,0)', zorder=3)
    ax.scatter([target_y], [target_p], color='red', marker='*', s=350, edgecolor='black', label='真实目标', zorder=3)
    
    ax.set_title(titles[i], fontsize=15, fontweight='bold', pad=10)
    ax.set_xlabel('水平方位角 Yaw (度)', fontsize=12)
    if i == 0: ax.set_ylabel('垂直高度角 Pitch (度)', fontsize=12)
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')
    
    cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.046, pad=0.15)
    cbar.set_label('时间 (秒)', fontsize=11)

plt.suptitle(f'单次头部探索轨迹对比 (目标编号: {TARGET_TO_OBSERVE} | Yaw:{target_y}°, Pitch:{target_p}°)', 
             fontsize=18, fontweight='bold', y=1.05)

plt.savefig(f'Fig3_Trajectory_Target{TARGET_TO_OBSERVE}.png', dpi=300, bbox_inches='tight')
plt.show()