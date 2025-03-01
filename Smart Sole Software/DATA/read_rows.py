import pandas as pd
import numpy as np

file = pd.read_csv("TrainingSet.csv", header=None)

l_column = file.iloc[:, -1]
matrix = []

for index, row in file.iterrows():
    if row.iloc[-1] == 2:

        matrix.append(row.iloc[:-1].values)

for row in matrix:
    print(row)