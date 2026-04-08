import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# 0. 基础设置与中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 假设您已经像之前的代码一样处理出了 pitch_0_df (Pitch == 0 的净时间数据)
# 这里为了方便您直接运行，我写了读取和处理的简写版（请确保在您的文件夹下运行）
folder_path = "E:\\Project\\6CCE3EEP_FYP\\Analysing\\PC\\Adjusted0_Data\\"
folder_path_train = "E:\\Project\\6CCE3EEP_FYP\\"

df_all = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(folder_path, "*_TrackingData_*.csv"))], ignore_index=True)

try:
    train_baseline_df = pd.read_csv(os.path.join(folder_path_train, 'Training_Converged_Results.csv'))
except FileNotFoundError:
    train_baseline_df = pd.read_csv(os.path.join(folder_path_train, 'Training_Converged_Results_Max20.csv'))

baseline_dict = dict(zip(train_baseline_df['ParticipantID'], train_baseline_df['Converged_Time_Last5_sec']))

rt_df = df_all.groupby(['ParticipantID', 'Phase', 'Trial', 'isCorrect', 'TargetNumber'])['TimeStamp_sec'].max().reset_index()
rt_df.rename(columns={'TimeStamp_sec': 'ReactionTime'}, inplace=True)
rt_df_correct = rt_df[(rt_df['isCorrect'] == 1) & (rt_df['ReactionTime'] <= 10.0)].copy()

rt_df_correct['BaselineTime'] = rt_df_correct['ParticipantID'].map(baseline_dict)
rt_df_correct = rt_df_correct.dropna(subset=['BaselineTime'])
rt_df_correct['Adjusted_RT'] = rt_df_correct['ReactionTime'] - rt_df_correct['BaselineTime']

azimuth_map = [0, 45, 90, 135, 180, -135, -90, -45]
elevation_map = [-30, 0, 30, 60]
rt_df_correct['Target_Yaw'] = rt_df_correct['TargetNumber'].apply(lambda x: azimuth_map[int(x) // 4])
rt_df_correct['Target_Pitch'] = rt_df_correct['TargetNumber'].apply(lambda x: elevation_map[int(x) % 4])

pitch_0_df = rt_df_correct[rt_df_correct['Target_Pitch'] == 0].copy()

# ==========================================
# 2. 极坐标图 (雷达图) 核心绘制逻辑
# ==========================================
# 计算每个角度的平均时间
spatial_means = pitch_0_df.groupby(['Phase', 'Target_Yaw'])['Adjusted_RT'].mean().reset_index()

# 将负角度转换为 0-360 的正角度，便于在圆盘上排序
spatial_means['Yaw_360'] = spatial_means['Target_Yaw'].apply(lambda x: x if x >= 0 else x + 360)

# 创建画布，必须声明 projection='polar'
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

phases = ['HRTF_A', 'HRTF_B']
colors = ['#4C72B0', '#DD8452']

for i, phase in enumerate(phases):
    phase_data = spatial_means[spatial_means['Phase'] == phase].sort_values('Yaw_360')
    
    # 获取角度和数值
    angles = np.deg2rad(phase_data['Yaw_360'].values)
    values = phase_data['Adjusted_RT'].values
    
    # 【核心】：为了让雷达图闭合，需要把第一个点的数据复制并添加到最后
    angles = np.concatenate((angles, [angles[0]]))
    values = np.concatenate((values, [values[0]]))
    
    # 绘制折线与填充
    ax.plot(angles, values, color=colors[i], linewidth=2.5, linestyle='solid', label=phase)
    ax.fill(angles, values, color=colors[i], alpha=0.2)

# =================美化极坐标图=================
# 旋转坐标系，让 0 度（正前方）指向上方，符合人类直觉
ax.set_theta_zero_location('N')
# 设置顺时针方向递增 (Right=90, Bottom=180, Left=270)
ax.set_theta_direction(-1)

# 自定义刻度标签
ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
ax.set_xticklabels(['0°\n(Front)', '45°', '90°\n(Right)', '135°', '180°\n(Rear)', '-135°', '-90°\n(Left)', '-45°'], fontsize=11)

plt.title('Horizontal Plane (0° Elevation) Localization Profile', fontsize=16, fontweight='bold', pad=30)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig('Fig_Pitch0_PolarPlot.png', dpi=300)
print("✅ 极坐标雷达图已生成！")
plt.show()