import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# 0. 读取和整合数据
# ==========================================
# 自动读取当前目录下所有包含 "TrainingData" 的 CSV 文件
path = "E:\\Project\\6CCE3EEP_FYP\\Analysing\\PC"
training_files = glob.glob(os.path.join(path, '*_TrainingData_*.csv'))
df_train = pd.concat([pd.read_csv(os.path.join(path, f)) for f in training_files])
df_train = df_train[df_train['AttemptNum'] <= 20]

# ==========================================
# 1. 第一部分：绘制反应时间趋势图
# ==========================================
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

# 绘制折线图
sns.lineplot(
    data=df_train,
    x='AttemptNum',
    y='ReactionTime_sec',
    hue='ParticipantID',
    marker='o',
    linewidth=2,
    palette="tab10"
)

# 完善图表标签与格式
plt.title('Reaction Time Trend Across Training Attempts', fontsize=16)
plt.xlabel('Attempt Number', fontsize=14)
plt.ylabel('Reaction Time (Seconds)', fontsize=14)
plt.xticks(np.arange(1, df_train['AttemptNum'].max() + 1, 1))
plt.legend(title='Participant ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# 导出图表图片
plt.savefig('Training_ReactionTime_Trend.png', dpi=300, bbox_inches='tight')
print("✅ 图表已成功生成并保存为 'Training_ReactionTime_Trend.png'。")

# ==========================================
# 2. 第二部分：计算最终趋近数值并导出表格
# ==========================================
results = []

for pid, group in df_train.groupby('ParticipantID'):
    # 确保按轮次升序排序，避免数据乱序导致抓取错误
    group = group.sort_values('AttemptNum')
    
    total_attempts = group['AttemptNum'].max()
    last_3_avg = group.tail(3)['ReactionTime_sec'].mean()
    last_5_avg = group.tail(5)['ReactionTime_sec'].mean()
    min_time = group['ReactionTime_sec'].min()
    
    results.append({
        'ParticipantID': pid,
        'Total_Attempts': total_attempts,
        'Converged_Time_Last3_sec': round(last_3_avg, 4),
        'Converged_Time_Last5_sec': round(last_5_avg, 4),
        'Best_Time_sec': round(min_time, 4)
    })

# 将计算结果保存为CSV文件
df_results = pd.DataFrame(results)
df_results.to_csv('Training_Converged_Results.csv', index=False)

print("✅ 趋近数值计算完成，已保存为 'Training_Converged_Results.csv'。")