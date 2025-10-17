import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

file_path = 'results_4.0.xlsx'
df = pd.read_excel(file_path)

required_columns = ['story_length', 'qa_order', 'is_correct']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Excel 文件中缺少必要的列：'{col}'，请检查列名。现有列有：{df.columns.tolist()}")

if len(df) != 180:
    raise ValueError(f"数据行数不是 180，实际是 {len(df)} 行，请确保输入数据正确。")

df['group_key'] = df['story_length'].astype(str) + df['qa_order'].astype(str)
group_sizes = df['group_key'].value_counts().sort_index()


def convert_to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.upper() == 'TRUE'
    if isinstance(x, int) or isinstance(x, float):
        return bool(x)
    return False


df['is_correct_bool'] = df['is_correct'].apply(convert_to_bool)

group_stats = df.groupby('group_key')['is_correct_bool'].agg(
    true_count=lambda x: (x == True).sum(),
    total_count=lambda x: len(x),
).reset_index()

group_stats['true_ratio'] = group_stats['true_count'] / group_stats['total_count']

x_labels = group_stats['group_key'].tolist()
y_values = group_stats['true_ratio'].tolist()
print(y_values)
plt.figure(figsize=(10, 5))
plt.plot(x_labels, y_values, marker='o', linestyle='-', color='b', markersize=8)

plt.title('Claude_4.5', fontsize=14)
plt.xlabel('story_length & qa_order 组合 (如 11=story=1,qa=1)', fontsize=12)
plt.ylabel('正确率', fontsize=12)
plt.ylim(0, 1.2)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, txt in enumerate(y_values):
    plt.annotate(f"{txt:.2f}", (x_labels[i], y_values[i]), textcoords="offset points", xytext=(0, 10), ha='center')

plt.tight_layout()
plt.show()
