import pandas as pd

models = []
execution_times = []
with open("bench.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        model, execution_time = line.split(" ")
        
        models.append(model)
        execution_times.append(float(execution_time))

df = pd.DataFrame([], columns=["model", "execution time"])

df["model"] = models
df["execution time"] = execution_times

df = df[df["execution time"] < 0.016]

print(df)

total_df = pd.read_csv("results-imagenet.csv")
total_df["execution time"] = 0

for index, row in df.iterrows():
    total_df.loc[total_df["model"] == row["model"], "execution time"] = row["execution time"]

total_df = total_df[total_df["execution time"] != 0]

print(total_df.loc[total_df["model"] == "cspdarknet53", "top1"])

total_df = total_df[total_df["top1"] >= 74]
total_df.reset_index(drop=True, inplace=True)

total_df
print(total_df.loc[:, ["model", "top1", "param_count", "img_size", "execution time"]])

total_df = total_df.loc[:, ["model", "top1", "param_count", "img_size", "execution time"]]
total_df = total_df.sort_values(by=['execution time'], axis=0)

markdown_table = total_df.to_markdown()
print(markdown_table)
