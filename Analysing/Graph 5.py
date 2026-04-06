import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# 3. 循环读取并合并
df_list = []
for file in file_list:
    temp_df = pd.read_csv(file)
    df_list.append(temp_df)

# pd.concat 就像是把所有表格上下拼接在一起
df_all = pd.concat(df_list, ignore_index=True)

print(f"成功合并了 {len(file_list)} 个被试的数据！共 {len(df_all)} 行。")

# 接下来，你就可以把 df_all 直接当成之前代码里的 df 来画图了！


# 解码目标的真实绝对坐标
azimuth_map = [0, 45, 90, 135, 180, -135, -90, -45]
elevation_map = [-30, 0, 30, 60]
df_all['Target_Yaw'] = df_all['TargetNumber'].apply(lambda x: azimuth_map[int(x) // 4])
df_all['Target_Pitch'] = df_all['TargetNumber'].apply(lambda x: elevation_map[int(x) % 4])

# ==========================================
# 2. 计算绝对误差与【时间分箱】
# ==========================================
def normalize_angle(angle):
    angle = angle % 360
    return np.where(angle > 180, angle - 360, angle)

# 我们关心的是“偏离了多少度”，所以要加上 np.abs() 取绝对值
df_all['Yaw_Error_Abs'] = np.abs(normalize_angle(df_all['Yaw_deg'] - df_all['Target_Yaw']))
df_all['Pitch_Error_Abs'] = np.abs(df_all['Pitch_deg'] - df_all['Target_Pitch'])

# 【核心：时间分箱 Time Binning】
# 把连续的时间戳四舍五入到最近的 0.05 秒（相当于 20Hz 采样率），以便对齐所有试次的数据
BIN_SIZE = 0.05  
df_all['Time_Bin'] = (df_all['TimeStamp_sec'] / BIN_SIZE).round() * BIN_SIZE

# 【截断长尾】由于找得快的人 1.5 秒就结束了，3秒以后的数据只剩下“找得慢的极端情况”
# 为了保证平均曲线的平滑和客观，我们只画出前 3.0 秒（或 2.5 秒）的收敛过程
MAX_TIME_TO_PLOT = 3.0
plot_data = df_all[df_all['Time_Bin'] <= MAX_TIME_TO_PLOT]

# ==========================================
# 3. 开始绘图 (带置信区间的折线图)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

# 定义颜色，保持和之前图表的一致性
palette = {'HRTF_A': '#4C72B0', 'HRTF_B': '#DD8452'}

# ----------------- 左图：水平角度 (Yaw) 收敛 -----------------
sns.lineplot(
    data=plot_data, 
    x='Time_Bin', 
    y='Yaw_Error_Abs', 
    hue='Phase', 
    palette=palette,
    linewidth=2.5,
    errorbar=('ci', 95), # 画出 95% 置信区间阴影 (老版本seaborn可能是 ci=95)
    ax=axes[0]
)
axes[0].set_title('水平方向 (Yaw) 角度收敛曲线', fontsize=15, fontweight='bold', pad=10)
axes[0].set_ylabel('绝对角度误差 (度)', fontsize=12)

# ----------------- 右图：垂直高度 (Pitch) 收敛 -----------------
sns.lineplot(
    data=plot_data, 
    x='Time_Bin', 
    y='Pitch_Error_Abs', 
    hue='Phase', 
    palette=palette,
    linewidth=2.5,
    errorbar=('ci', 95),
    ax=axes[1]
)
axes[1].set_title('垂直方向 (Pitch) 角度收敛曲线', fontsize=15, fontweight='bold', pad=10)
axes[1].set_ylabel('') # 共用语意，省略 Y 轴标签

# ----------------- 全局美化 -----------------
for ax in axes:
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_xlim(0, MAX_TIME_TO_PLOT)
    ax.set_ylim(bottom=0) # 误差最小就是 0 度，也就是完美对齐
    
    # 画一条 Y=0 的绿色虚线，代表“完美对齐目标”的理想终点
    ax.axhline(0, color='green', linestyle='--', linewidth=1.5, alpha=0.8, label='完美对齐 (0度)')
    
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # 调整图例
    handles, labels = ax.get_legend_handles_labels()
    # 过滤掉不需要的图例项
    if 'Phase' in labels:
        idx = labels.index('Phase')
        handles.pop(idx)
        labels.pop(idx)
    ax.legend(handles=handles, labels=labels, loc='upper right')

plt.suptitle('头部追踪运动学：目标角度误差时间收敛对比', fontsize=18, fontweight='bold', y=1.05)

plt.savefig('Fig5_Time_Series_Convergence.png', dpi=300, bbox_inches='tight')
plt.show()