import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import glob
import os
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 仅读取 Tracking 数据
# ==========================================
folder_path = "E:\\Project\\6CCE3EEP_FYP\\Analysing\\PC\\Adjusted0_Data\\"
file_list = glob.glob(os.path.join(folder_path, "*_TrackingData_*.csv"))

df_all = pd.concat([pd.read_csv(f) for f in file_list], ignore_index=True)
print(f"成功合并了 {len(file_list)} 个 Tracking 数据！")

# ==========================================
# 2. 数据处理：提取原始反应时间 (Raw Reaction Time)
# ==========================================
rt_df = df_all.groupby(['ParticipantID', 'Phase', 'Trial', 'isCorrect'])['TimeStamp_sec'].max().reset_index()
rt_df.rename(columns={'TimeStamp_sec': 'ReactionTime'}, inplace=True)

# 仅保留答对的试次
rt_df_correct = rt_df[rt_df['isCorrect'] == 1].copy()

# ==========================================
# 3. 数据过滤：剔除极端离群值 (基于原始 ReactionTime)
# ==========================================
original_count = len(rt_df_correct)
Q1 = rt_df_correct['ReactionTime'].quantile(0.25)
Q3 = rt_df_correct['ReactionTime'].quantile(0.75)
IQR = Q3 - Q1

# 下限设为0即可，因为真实时间不可能为负
lower_bound = max(0, Q1 - 1.5 * IQR)
upper_bound = Q3 + 1.5 * IQR

rt_df_filtered = rt_df_correct[
    (rt_df_correct['ReactionTime'] >= lower_bound) & 
    (rt_df_correct['ReactionTime'] <= upper_bound)
]

dropped_count = original_count - len(rt_df_filtered)
print(f"【数据清洗】已剔除 {dropped_count} 条离群数据 (上限为 {upper_bound:.2f}s)。")

# ==========================================
# 4. 开始绘图 (小提琴图 + 蜂群散点图)
# ==========================================
plt.figure(figsize=(9, 6))
sns.set_theme(style="whitegrid") 

sns.violinplot(
    x='Phase', 
    y='ReactionTime', 
    data=rt_df_filtered, 
    inner="box",         
    palette=["#4C72B0", "#DD8452"], 
    alpha=0.8
)

sns.swarmplot(
    x='Phase', 
    y='ReactionTime', 
    data=rt_df_filtered, 
    color='black', 
    alpha=0.6, 
    size=4               
)

# =================美化细节=================
plt.title('Raw Search Time Distribution Comparison', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('HRTF Condition', fontsize=14)
plt.ylabel('Search Time (seconds)', fontsize=14)

plt.xticks(ticks=[0, 1], labels=['Generic HRTF (A)', 'Individualised HRTF (B)'], fontsize=12)

# 强制纵轴 0.5s 一格
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.5))
plt.yticks(fontsize=11)

plt.tight_layout()
plt.savefig('Ad_ReactionTime_Violinplot.png', dpi=300, bbox_inches='tight')
print("✅ 基于原始时间的图表已成功生成并保存！")
plt.show()