import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Login using e.g. `huggingface-cli login` to access this dataset
splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/ailsntua/QEvasion/" + splits["train"])

label_col = 'clarity_label'
columns_dict = df[label_col].value_counts()

plt.figure(figsize=(12, 10))
plt.bar(columns_dict.keys(), columns_dict.values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
plt.title('Clarity Label distribution in CLARITY')
plt.xlabel('Answer type')
plt.ylabel('Counts')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation = 30)
plt.show()

label_col = 'evasion_label'
columns_dict = df[label_col].value_counts()

plt.figure(figsize=(12, 10))
plt.bar(columns_dict.keys(), columns_dict.values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
plt.title('Evasion Label distribution in CLARITY')
plt.xlabel('Answer type')
plt.ylabel('Counts')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation = 30)
plt.show()


heatmap_data = pd.crosstab(df['president'], df[label_col], normalize='index')

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar_kws={'label': 'Rate (0-1)'})

plt.title('Label distribution per president')
plt.xlabel('Answer type')
plt.ylabel('President')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()

