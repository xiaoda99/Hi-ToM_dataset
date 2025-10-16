import json
import pandas as pd

# 从 json 文件读取
with open("data1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 转为 DataFrame
df = pd.DataFrame(data)

# 保存为表格
df.to_excel("data_check.xlsx", index=False)