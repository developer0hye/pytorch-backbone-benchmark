import pandas as pd
import numpy as np

models = []
execution_times = []
with open("bench.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        model, execution_time = line.split(" ")
        
        models.append(model)
        execution_times.append(np.round(float(execution_time), 4))

execution_times = np.array(execution_times)

df = pd.DataFrame([], columns=["model", "execution time(sec)"])

df["model"] = models
df["execution time(sec)"] = execution_times
df["fps"] = np.floor(1/execution_times)

total_df = pd.read_csv("pytorch-image-models/results/results-imagenet.csv")
total_df["execution time(sec)"] = 0
total_df["fps"] = 0

for index, row in df.iterrows():
    total_df.loc[total_df["model"] == row["model"], "execution time(sec)"] = row["execution time(sec)"]
    total_df.loc[total_df["model"] == row["model"], "fps"] = row["fps"]

total_df = total_df[total_df["execution time(sec)"] != 0]
total_df = total_df.loc[:, ["model", "top1", "param_count", "execution time(sec)", "fps"]]
total_df = total_df.sort_values(by=['execution time(sec)'], axis=0)
total_df.reset_index(drop=True, inplace=True)

total_df.to_csv("benchmark.csv")

markdown_table = total_df.to_markdown()
print(markdown_table)
