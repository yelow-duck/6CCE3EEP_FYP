import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import glob
import os
import warnings

# 忽略 Seaborn 的一些版本警告
warnings.filterwarnings("ignore")

# ==========================================
# 0. 基础设置与中文字体支持
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 读取数据 (包含 Tracking 和 Training 基线)
# ==========================================
# 1.1 读取 Tracking 数据
folder_path = "E:\\Project\\6CCE3EEP_FYP\\Analysing\\PC\\"
file_list = glob.glob(os.path.join(folder_path, "*_TrackingData_*.csv"))

df_list = []
for file in file_list:
    temp_df = pd.read_csv(file)
    df_list.append(temp_df)

df_all = pd.concat(df_list, ignore_index=True)
print(f"成功合并了 {len(file_list)} 个 Tracking 数据！共 {len(df_all)} 行。")

# 1.2 读取 Training 趋近结果 (提取基线)
# 修复了之前的路径问题，确保它去 PC 文件夹下找这个文件
folder_path_train = "E:\\Project\\6CCE3EEP_FYP\\"
baseline_file_path = os.path.join(folder_path_train, 'Training_Converged_Results.csv')
train_baseline_df = pd.read_csv(baseline_file_path)

# 我们采用“最后5次的平均趋近时间”作为该被试的个人生理基线
baseline_dict = dict(zip(train_baseline_df['ParticipantID'], train_baseline_df['Converged_Time_Last5_sec']))

# ==========================================
# 2. 数据处理：计算调整后的反应时间 (Adjusted Reaction Time)
# ==========================================
# 提取反应时间
rt_df = df_all.groupby(['ParticipantID', 'Phase', 'Trial', 'isCorrect'])['TimeStamp_sec'].max().reset_index()
rt_df.rename(columns={'TimeStamp_sec': 'ReactionTime'}, inplace=True)

# 过滤答对的试次
rt_df_correct = rt_df[rt_df['isCorrect'] == 1].copy()

# 【核心步骤】：映射被试的 Training 基线，并做减法
rt_df_correct['BaselineTime'] = rt_df_correct['ParticipantID'].map(baseline_dict)
rt_df_correct = rt_df_correct.dropna(subset=['BaselineTime'])
rt_df_correct['Adjusted_RT'] = rt_df_correct['ReactionTime'] - rt_df_correct['BaselineTime']

# ==========================================
# 2.5 数据过滤：剔除极端离群值 (基于 Adjusted_RT)
# ==========================================
original_count = len(rt_df_correct)

Q1 = rt_df_correct['Adjusted_RT'].quantile(0.25)
Q3 = rt_df_correct['Adjusted_RT'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

rt_df_filtered = rt_df_correct[
    (rt_df_correct['Adjusted_RT'] >= lower_bound) & 
    (rt_df_correct['Adjusted_RT'] <= upper_bound)
]

dropped_count = original_count - len(rt_df_filtered)
print(f"【数据清洗】已剔除 {dropped_count} 条离群数据。")

# ==========================================
# 3. 开始绘图 (小提琴图 + 蜂群散点图)
# ==========================================
plt.figure(figsize=(9, 6))
sns.set_theme(style="whitegrid") # 使用带网格的白底，更适合看清刻度

# 3.1 画小提琴图底色
sns.violinplot(
    x='Phase', 
    y='Adjusted_RT', 
    data=rt_df_filtered, 
    inner="box",         # 内部保留迷你箱线图看四分位数
    palette=["#4C72B0", "#DD8452"], 
    alpha=0.8
)

# 3.2 画内部的散点 (蜂群图 Swarmplot)
sns.swarmplot(
    x='Phase', 
    y='Adjusted_RT', 
    data=rt_df_filtered, 
    color='black', 
    alpha=0.6, 
    size=4               # 稍微调小了点大小，防止您6个人的数据点重叠太厉害
)

# ==========================================
# 4. 图表美化与细节设置
# ==========================================

plt.xlabel('Experiment Condition', fontsize=13)
plt.ylabel('Adjusted Time (s)', fontsize=13)

# 更改 X 轴标签
plt.xticks(ticks=[0, 1], labels=['HRTF A', 'HRTF B'], fontsize=12)

# 【核心要求】：强制纵轴 (Y轴) 的刻度间隔为 0.5 秒
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.5))
plt.yticks(fontsize=11)

# 保存与显示
plt.tight_layout()
plt.savefig('Figure 1 Tracking_Adjusted_RT_Violinplot.png', dpi=300, bbox_inches='tight')
print("✅ 图表已成功生成并保存为 'Figure 1 Tracking_Adjusted_RT_Violinplot.png'。")
plt.show()