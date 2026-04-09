import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import glob
import os
import warnings

import scipy.stats as stats

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 读取【截断后】的纯净实验数据
# ==========================================
folder_path = "E:\\Project\\6CCE3EEP_FYP\\Analysing\\PC\\Adjusted0_Data\\"
file_list = glob.glob(os.path.join(folder_path, "*_TrackingData_*.csv"))

df_all = pd.concat([pd.read_csv(f) for f in file_list], ignore_index=True)
print(f"成功读取了 {len(file_list)} 个被试的截断数据！")

# ==========================================
# 2. 数据处理：计算每位被试的中位数 (Median)
# ==========================================
# 第一步：获取每个纯净试次的最终寻找时间 (取每个 trial 截断后的最大时间)
rt_df = df_all[df_all['isCorrect'] == 1].groupby(['ParticipantID', 'Phase', 'Trial'])['TimeStamp_sec'].max().reset_index()
rt_df.rename(columns={'TimeStamp_sec': 'SearchTime'}, inplace=True)

# 第二步：【核心】计算每个 Participant 在每种 Phase 下的中位数
median_df = rt_df.groupby(['ParticipantID', 'Phase'])['SearchTime'].median().reset_index()

print("\n每位参与者的中位数概览：")
print(median_df.pivot(index='ParticipantID', columns='Phase', values='SearchTime'))

# ==========================================
# 3. 开始绘图 (小提琴底图 + 散点 + 个人变化连线)
# ==========================================
plt.figure(figsize=(8, 6))
sns.set_theme(style="ticks")

# 1. 画小提琴图 (设为半透明作为背景分布)
sns.violinplot(
    x='Phase', 
    y='SearchTime', 
    data=median_df, 
    inner=None, # 取消内部箱线，保持画面整洁 
    palette=["#4C72B0", "#DD8452"], 
    alpha=0.4   # 透明度调高，突出前面的连线
)

# 2. 画出每个人的中位数散点 (因为点很少，把点放大)
sns.stripplot(
    x='Phase', 
    y='SearchTime', 
    data=median_df, 
    color='black', 
    size=8, 
    alpha=0.8,
    zorder=10
)

# 3. 【学术高阶作图：画个体变化连线】
# 将数据重塑为宽表，方便提取同一人的 A 和 B
pivot_df = median_df.pivot(index='ParticipantID', columns='Phase', values='SearchTime')

# 遍历每一位参与者，画线连接他们在 A 和 B 下的表现
for idx, row in pivot_df.iterrows():
    # x坐标: HRTF_A 是 0, HRTF_B 是 1
    # y坐标: 对应的中位数时间
    plt.plot([0, 1], [row['HRTF_A'], row['HRTF_B']], color='gray', linestyle='-', linewidth=2, alpha=0.6, zorder=5)

# =================美化细节=================
plt.title('Individual Median Search Time: With headphone and No device HRTF', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('HRTF Condition', fontsize=13)
plt.ylabel('Median Search Time (seconds)', fontsize=13)

plt.xticks(ticks=[0, 1], labels=['With headphone', 'No device'], fontsize=12)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.5))

sns.despine(top=True, right=True)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('Median_Paired_Violinplot.png', dpi=300, bbox_inches='tight')
print("\n✅ 中位数个体连线图已成功生成并保存！")
plt.show()


# 提取你之前 pivot_df 里的两组中位数
group_A = pivot_df['HRTF_A']
group_B = pivot_df['HRTF_B']

# 计算配对差值
differences = group_B - group_A

# 执行 Shapiro-Wilk 正态性检验
stat_sw, p_sw = stats.shapiro(differences)

print(f"Shapiro-Wilk 统计量 (W): {stat_sw:.4f}")
print(f"P 值 (p-value): {p_sw:.4f}")
if p_sw < 0.05:
    print("结论：差值【不符合】正态分布 (直接证明必须用 Wilcoxon)")
else:
    print("结论：差值【符合】正态分布 (但因为只有8个人，检验效力太弱，依然应该用 Wilcoxon)")


# 假设你已经有了上一段代码生成的 median_df
# 将数据重塑，确保同一被试的 A 和 B 数据是严格对齐的 (Paired)
pivot_df = median_df.pivot(index='ParticipantID', columns='Phase', values='SearchTime')

# 提取两组中位数
group_A = pivot_df['HRTF_A']
group_B = pivot_df['HRTF_B']

# 执行 Wilcoxon 符号秩检验
stat, p_value = stats.wilcoxon(group_A, group_B)

print(f"Wilcoxon 统计量 (W): {stat}")
print(f"P 值 (p-value): {p_value:.4f}")

if p_value < 0.05:
    print("✅ 结论：在统计学上，HRTF B 的搜索时间显著短于 HRTF A (p < 0.05)。")
else:
    print("❌ 结论：两者差异在统计学上不显著 (p >= 0.05)。")