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
# 2. 【核心算法】计算相对误差角度 (以目标为绝对中心点 0,0)
# ==========================================
# 计算水平误差 (Yaw_Error) 并将其归一化到 -180 到 180 度之间 (防止出现 350度 这种数值)
def normalize_angle(angle):
    angle = angle % 360
    return np.where(angle > 180, angle - 360, angle)

df_all['Yaw_Error'] = normalize_angle(df_all['Yaw_deg'] - df_all['Target_Yaw'])
# 垂直误差直接相减 (VR里的 Pitch 一般限制在 -90 到 90，不会跨越 360)
df_all['Pitch_Error'] = df_all['Pitch_deg'] - df_all['Target_Pitch']

# ==========================================
# 3. 开始绘图 (双子图对比 2D 核密度分布)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
phases = ['HRTF_A', 'HRTF_B']
titles = ['HRTF A: 全局注视点热力分布', 'HRTF B: 全局注视点热力分布']

# 设置绘图视窗的范围（我们只聚焦目标周围 ±90 度的区域，看火球有多大）
LIMIT_X = 90
LIMIT_Y = 60

for i, phase in enumerate(phases):
    ax = axes[i]
    phase_data = df_all[df_all['Phase'] == phase]
    
    # 过滤掉超出视窗的极端离群点，让 KDE 算法算得更准
    plot_data = phase_data[(np.abs(phase_data['Yaw_Error']) <= LIMIT_X) & 
                           (np.abs(phase_data['Pitch_Error']) <= LIMIT_Y)]
    
    # 画 2D KDE (Kernel Density Estimate) 核密度热力图
    # cmap 推荐使用 'rocket' (黑-红-黄) 或 'magma' (黑-紫-橙-黄)，视觉冲击力极强
    sns.kdeplot(
        x=plot_data['Yaw_Error'], 
        y=plot_data['Pitch_Error'],
        cmap='magma',     # 火山岩浆配色
        fill=True,        # 填充颜色
        thresh=0.05,      # 过滤掉最外围 5% 的零散噪点，让图更干净
        levels=20,        # 等高线的细腻程度
        ax=ax
    )
    
    # 在中心 (0,0) 画一个极其醒目的十字准星，代表“真正的目标在这里”
    ax.axhline(0, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(0, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.scatter([0], [0], color='cyan', marker='+', s=200, linewidth=2, label='目标真实位置')
    
    # 坐标轴与标题设置
    ax.set_title(titles[i], fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('水平视线误差 Yaw Error (度)', fontsize=12)
    if i == 0:
        ax.set_ylabel('垂直视线误差 Pitch Error (度)', fontsize=12)
    
    ax.set_xlim(-LIMIT_X, LIMIT_X)
    ax.set_ylim(-LIMIT_Y, LIMIT_Y)
    ax.legend(loc='upper right', framealpha=0.8)

plt.suptitle('空间听觉引导精度对比：相对注视误差热力图', fontsize=18, fontweight='bold', y=1.05)

# 保存高刷大图
plt.savefig('Fig4_Gaze_Density_Heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()