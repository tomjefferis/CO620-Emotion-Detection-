import pandas as pd


dataset = pd.read_csv("./processed_data/completeDatanoeye.csv")

dataset.loc[dataset["labels"] == 1, "labels"] = 0
dataset.loc[dataset["labels"] == 2, "labels"] = 1
dataset.loc[dataset["labels"] == 3, "labels"] = 1

dataset.to_csv("./processed_data/completeDatabinarynoeye.csv")
