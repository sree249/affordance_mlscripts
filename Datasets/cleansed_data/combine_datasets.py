#combine datasets

import pandas as pd

data1 = pd.read_csv("Affordance_November4_alldatacleansed_validated_new_data.csv")
data2 = pd.read_csv("Affordance_November 5 2018cleansed_validated_new_data.csv")
data3 = pd.read_csv("Affordance_November 6 2018cleansed_validated_new_data.csv")
data4 = pd.read_csv("Affordance_November 8 2018cleansed_validated_new_data.csv")
data5 =pd.read_csv("Affordance_November 9 2018cleansed_validated_new_data.csv")
data6 = pd.read_csv("Affordance_November 10 2018cleansed_validated_new_data.csv")
data7  = pd.read_csv("Affordance_November 12 2018cleansed_validated_new_data.csv")
data8 = pd.read_csv("Affordance_November 13 2018cleansed_validated_new_data.csv")
data9 = pd.read_csv("Affordance_November 15 2018cleansed_validated_new_data.csv")



final_data = pd.concat([data1,data2,data3,data4,data5,data6,data7,data8,data9])

final_data.to_csv("Combined_data_November16.csv")