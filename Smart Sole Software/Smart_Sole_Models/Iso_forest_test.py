import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest as iso
import pandas as pd

Walking_data_reg = pd.read_csv("Iso_walking.csv")
Standing_data_reg = pd.read_csv("Iso_standing.csv")
Sitting_data_reg = pd.read_csv("Iso_sitting.csv")
Limping_data_reg = pd.read_csv("Iso_limping.csv")
Heel_av_st_data_reg = pd.read_csv("Iso_Heel_Av_st.csv")
Heel_av_dy_data_reg = pd.read_csv("Iso_Heel_Av_dy.csv")
Lat_pre_st_data_reg = pd.read_csv("Iso_Lat_pre_st.csv")
Lat_pre_dy_data_reg = pd.read_csv("Iso_Lat_pre_dy.csv")

Input_data = Walking_data_reg.to_numpy()


New_row = np.array([877,4095,36,4095,4095,4095,4095,4095,2840,285,4095,24,40,4095,3379,4095,4095,4095,4095,4095,2812,52,4095,26,796,4095,1848,4095,4095,4095])
Input_data = np.vstack([Input_data, New_row])
iso_model = iso()
iso_model.fit(Input_data)
predict = iso_model.predict(Input_data)
scores = iso_model.score_samples(Input_data)

added_index = Input_data.shape[0] -1
is_anomaly = predict[added_index] == -1
added_score = scores[added_index]

print("Predictions (1=Normal, -1=Anomaly):", predict)
print("Anomaly Scores:", scores)
print(f"Added Dataset (Row {added_index}): {Input_data[added_index]}")
print(f"Is the added dataset an anomaly? {'Yes' if is_anomaly else 'No'}")
print(f"Anomaly Score of the added dataset: {added_score}")
plt.hist(scores, bins=20)
plt.title("Anomaly Scores Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()