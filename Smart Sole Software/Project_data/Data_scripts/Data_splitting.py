import pandas as pd

file = pd.read_csv("LATERAL_ARCH_PRESSURE_DYNAMIC.csv", header=None)

chunk = []  
for index, row in file.iterrows():
    chunk.append(row.values.tolist())
    if (index + 1) % 10 == 0:  
        print(chunk)
        print("\n")  
        chunk = []  
if chunk:
    print(chunk)