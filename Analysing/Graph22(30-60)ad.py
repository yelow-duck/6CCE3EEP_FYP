import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 数据读取与预处理 (沿用之前的标准流程)
# ==========================================
folder_path = "E:\\Project\\6CCE3EEP_FYP\\Analysing\\PC\\Adjusted0_Data\\"
folder_path_train = "E:\\Project\\6CCE3EEP_FYP\\"

# 读 Tracking
df_all = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(folder_path, "*_TrackingData_*.csv"))], ignore_index=True)

# 读 Training 基线
try:
    train_baseline_df = pd.read_csv(os.path.join(folder_path_train, 'Training_Converged_Results.csv'))
except FileNotFoundError:
    train_baseline_df = pd.read_csv(os.path.join(folder_path_train, 'Training_Converged_Results_Max20.csv'))
baseline_dict = dict(zip(train_baseline_df['ParticipantID'], train_baseline_df['Converged_Time_Last5_sec']))

# 计算净时间与坐标解码
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

# 【核心过滤】：只提取 30度 和 60度 仰角的数据
pitch_30_60_df = rt_df_correct[rt_df_correct['Target_Pitch'].isin([30, 60])].copy()

# ==========================================
# 图表 1：宏观对比 - 交互效应箱线图 (Boxplot)
# ==========================================
plt.figure(figsize=(8, 6))
sns.set_theme(style="ticks")

sns.boxplot(
    data=pitch_30_60_df, 
    x='Target_Pitch', 
    y='Adjusted_RT', 
    hue='Phase',
    width=0.5,
    palette=['#4C72B0', '#DD8452'],
    showfliers=False,
    boxprops=dict(alpha=0.8)
)
sns.stripplot(
    data=pitch_30_60_df, 
    x='Target_Pitch', 
    y='Adjusted_RT', 
    hue='Phase',
    dodge=True, 
    color='black', 
    alpha=0.4, 
    jitter=0.1, 
    size=4
)

# 修复图例重复的问题
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:2], labels[:2], title='HRTF Condition', loc='upper left')

plt.title('Elevation Stress Test: 30° vs 60° Pitch', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Target Elevation (Pitch Degrees)', fontsize=13)
plt.ylabel('Adjusted Search Time (s)', fontsize=13)
sns.despine()
plt.tight_layout()
plt.savefig('Fig_Pitch_30vs60_Boxplot.png', dpi=300)
print("✅ 图表1：30° vs 60° 交互效应箱线图已生成！")
plt.close()

# ==========================================
# 图表 2：微观对比 - 双雷达极坐标图
# ==========================================
# 计算每个条件和角度的平均值
spatial_means = pitch_30_60_df.groupby(['Phase', 'Target_Pitch', 'Target_Yaw'])['Adjusted_RT'].mean().reset_index()
spatial_means['Yaw_360'] = spatial_means['Target_Yaw'].apply(lambda x: x if x >= 0 else x + 360)

fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': 'polar'})
phases = ['HRTF_A', 'HRTF_B']
elevations = [30, 60]
# 30度用绿色，60度用红色，方便对比严重程度
colors_pitch = {30: '#2ECC71', 60: '#E74C3C'} 

# 获取全局最大值以统一两张图的半径刻度
rmax = spatial_means['Adjusted_RT'].max() * 1.1

for i, phase in enumerate(phases):
    ax = axes[i]
    
    for pitch in elevations:
        subset = spatial_means[(spatial_means['Phase'] == phase) & (spatial_means['Target_Pitch'] == pitch)].sort_values('Yaw_360')
        angles = np.deg2rad(subset['Yaw_360'].values)
        values = subset['Adjusted_RT'].values
        
        # 闭合雷达图
        angles = np.concatenate((angles, [angles[0]]))
        values = np.concatenate((values, [values[0]]))
        
        ax.plot(angles, values, color=colors_pitch[pitch], linewidth=2.5, linestyle='solid', label=f'Elevation {pitch}°')
        ax.fill(angles, values, color=colors_pitch[pitch], alpha=0.15)
    
    # 极坐标美化
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, rmax)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(['0°\n(Front)', '45°', '90°\n(Right)', '135°', '180°\n(Rear)', '-135°', '-90°\n(Left)', '-45°'], fontsize=10)
    ax.set_title(f'{phase} Performance', fontsize=14, fontweight='bold', pad=20)
    
axes[1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title="Target Elevation")
plt.suptitle('Spatial Degradation at High Elevations: 30° vs 60°', fontsize=18, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('Fig_Pitch_30vs60_PolarPlot.png', dpi=300)
print("✅ 图表2：30° vs 60° 双雷达空间分布图已生成！")
plt.show()