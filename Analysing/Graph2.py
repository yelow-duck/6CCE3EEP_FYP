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
# 1. 读取数据
# ==========================================
# 1. 指定存放所有数据的文件夹路径 (可以用相对路径)
# 假设你把所有下载的 CSV 都放在了代码同级的一个叫 "Data" 的文件夹里
folder_path = "E:\\Project\\6CCE3EEP_FYP\\Analysing\\PC\\"

# 2. 自动抓取该文件夹下所有符合命名规律的 csv 文件
# 匹配你在 Unity 里设置的文件名格式：P001_TrackingData_xxx.csv
file_list = glob.glob(os.path.join(folder_path, "*_TrackingData_*.csv"))
# 只分析答对的试次，剔除乱找的数据

# 3. 循环读取并合并
df_list = []
for file in file_list:
    temp_df = pd.read_csv(file)
    df_list.append(temp_df)

# pd.concat 就像是把所有表格上下拼接在一起
df_all = pd.concat(df_list, ignore_index=True)

print(f"成功合并了 {len(file_list)} 个被试的数据！共 {len(df_all)} 行。")

# 接下来，你就可以把 df_all 直接当成之前代码里的 df 来画图了！


# ==========================================
# 2. 数据处理：提取反应时间并根据 32 宫格解码位置
# ==========================================
# 获取每个试次的最终反应时间
rt_df = df_all.groupby(['ParticipantID', 'Phase', 'Trial', 'isCorrect', 'TargetNumber'])['TimeStamp_sec'].max().reset_index()
rt_df.rename(columns={'TimeStamp_sec': 'ReactionTime'}, inplace=True)

# 过滤：只保留答对的试次，并剔除大于 10 秒的离群值
rt_df_correct = rt_df[(rt_df['isCorrect'] == 1) & (rt_df['ReactionTime'] <= 10.0)].copy()

# 【核心更新：应用你的真实 32 目标坐标系】
# 定义 8 个水平方位和 4 个垂直高度的数组
azimuth_map = [0, 45, 90, 135, 180, -135, -90, -45]
elevation_map = [-30, 0, 30, 60]

# 将 TargetNumber (0-31) 转换为真实的 Yaw 和 Pitch
rt_df_correct['Target_Yaw'] = rt_df_correct['TargetNumber'].apply(lambda x: azimuth_map[int(x) // 4])
rt_df_correct['Target_Pitch'] = rt_df_correct['TargetNumber'].apply(lambda x: elevation_map[int(x) % 4])

# 分组计算 HRTF_A 和 HRTF_B 在每个空间位置的“平均寻找时间”
spatial_rt = rt_df_correct.groupby(['Phase', 'Target_Yaw', 'Target_Pitch'])['ReactionTime'].mean().reset_index()

# ==========================================
# 3. 开始绘图 (双子图对比气泡图)
# ==========================================
# 获取全局最短和最长的时间，确保左右两张图的颜色和气泡大小共用同一套标准
vmin = spatial_rt['ReactionTime'].min()
vmax = spatial_rt['ReactionTime'].max()

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
phases = ['HRTF_A', 'HRTF_B']
titles = ['HRTF A 空间耗时分布', 'HRTF B 空间耗时分布']

# 为了让画出来的图符合人类直觉，强制规范 X轴(从左到右) 和 Y轴(从下到上) 的刻度顺序
x_order = [-135, -90, -45, 0, 45, 90, 135, 180]
y_order = [-30, 0, 30, 60]

for i, phase in enumerate(phases):
    ax = axes[i]
    phase_data = spatial_rt[spatial_rt['Phase'] == phase]
    
    # 绘制散点气泡图
    scatter = sns.scatterplot(
        x='Target_Yaw', y='Target_Pitch',
        size='ReactionTime', hue='ReactionTime',
        sizes=(150, 1500), # 气泡大小的变化范围，可根据画面拥挤程度调整
        alpha=0.8,
        palette='YlOrRd',  # 黄-橙-红渐变，红色代表耗时最久 (盲区)
        data=phase_data,
        hue_norm=(vmin, vmax), 
        size_norm=(vmin, vmax),
        ax=ax,
        legend=False 
    )
    
    ax.set_title(titles[i], fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('水平方位角 Yaw (度)', fontsize=12)
    
    # 强制设置 X 和 Y 轴刻度，展示你设计的所有可能点位
    ax.set_xticks(x_order)
    ax.set_yticks(y_order)
    
    # 将坐标轴范围向外扩宽一点，防止位于边缘的气泡（比如 180度 或 60高度）被画框切掉一半
    ax.set_xlim(-155, 200)
    ax.set_ylim(-45, 75)
    
    # 加入网格辅助线，便于对应具体坐标
    ax.grid(True, linestyle='--', alpha=0.6)

axes[0].set_ylabel('垂直高度角 Pitch (度)', fontsize=12)

# ==========================================
# 4. 添加全局统一的颜色图例 (Colorbar)
# ==========================================
sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
cbar.set_label('平均寻找时间 (秒)', fontsize=12, labelpad=10)

plt.suptitle('空间听觉定位耗时分布对比 (红色气泡越大代表耗时越久)', fontsize=18, fontweight='bold', y=1.05)

# 保存高刷大图
plt.savefig('Fig2_Spatial_BubblePlot.png', dpi=300, bbox_inches='tight')
plt.show()