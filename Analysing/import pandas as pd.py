import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# ==========================================
# 0. 基础设置与中文字体支持 (防止图表中文显示方块)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] # Windows 默认黑体
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

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


# ==========================================
# 2. 数据处理：提取反应时间 (Reaction Time)
# ==========================================
# 因为 DataLogger 是一帧帧记录的，所以每个 Trial 最大(最后)的时间戳，就是反应时间
rt_df = df_all.groupby(['ParticipantID', 'Phase', 'Trial', 'isCorrect'])['TimeStamp_sec'].max().reset_index()
rt_df.rename(columns={'TimeStamp_sec': 'ReactionTime'}, inplace=True)

# 过滤：通常我们只分析寻找正确的试次，排除掉瞎猜或者找错的试次
# isCorrect == 1 表示答对（根据你 DataLogger 里的逻辑，1为对，0为错）
rt_df_correct = rt_df[rt_df['isCorrect'] == 1]


# ==========================================
# 2.5 数据过滤：剔除极端离群值 (IQR 统计法)
# ==========================================
original_count = len(rt_df_correct)

# 计算 25% 分位数 (Q1) 和 75% 分位数 (Q3)
Q1 = rt_df_correct['ReactionTime'].quantile(0.25)
Q3 = rt_df_correct['ReactionTime'].quantile(0.75)
IQR = Q3 - Q1

# 定义正常范围的上下限 (学术界公认标准：超过 Q3 + 1.5倍IQR 算极端值)
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 因为反应时间不能为负数，下限可以直接设为防误触的 0.2 秒
lower_bound = max(0.2, lower_bound)

# 执行过滤
rt_df_filtered = rt_df_correct[
    (rt_df_correct['ReactionTime'] >= lower_bound) & 
    (rt_df_correct['ReactionTime'] <= upper_bound)
]

dropped_count = original_count - len(rt_df_filtered)
print(f"【数据清洗】IQR 上限计算为 {upper_bound:.2f} 秒。")
print(f"【数据清洗】已剔除 {dropped_count} 条离群数据。")

# ⚠️ 注意：同样记得把画图代码里的 data 改成 rt_df_filtered



# ==========================================
# 3. 开始绘图 (小提琴图 + 蜂群散点图)
# ==========================================
plt.figure(figsize=(9, 6))

# 画小提琴图底色 (inner="box" 会在内部画一个迷你的箱线图)
sns.violinplot(
    x='Phase', 
    y='ReactionTime', 
    data=rt_df_filtered, 
    inner="box", 
    palette=["#4C72B0", "#DD8452"], 
    alpha=0.8
)

# 画内部的散点 (展示每一个真实的数据点)
sns.swarmplot(
    x='Phase', 
    y='ReactionTime', 
    data=rt_df_filtered, 
    color='black', 
    alpha=0.6, 
    size=5
)

# ==========================================
# 4. 图表美化与细节设置
# ==========================================
plt.title('Reaction Time Distribution (Correct Trials)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Experiment (HRTF)', fontsize=13)
plt.ylabel('Reaction Time (s)', fontsize=13)

# 更改 X 轴的刻度标签，让它更易读
plt.xticks(ticks=[0, 1], labels=['HRTF A Algorithm', 'HRTF B Algorithm'], fontsize=12)
plt.yticks(fontsize=11)

# 添加横向网格线辅助阅读
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 移除顶部和右侧的边框线
sns.despine()

# 紧凑布局并保存高清晰度图片 (写进论文需要至少 300 dpi)
plt.tight_layout()
plt.savefig('Fig1_ReactionTime_Distribution.png', dpi=300)

# 显示图表
plt.show()