#trying pandas dataset concacts

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
# dataset1 = 'Datasets/raw datasets day 1-6/Affordance_Day1'
# file_name = dataset1+".csv"
# data1 = pd.read_csv(file_name)

# dataset2 = 'Datasets/raw datasets day 1-6/Affordance_Day2'
# file_name = dataset2+".csv"
# data2 = pd.read_csv(file_name)

# dataset2 = 'Datasets/raw datasets day 1-6/Affordance_Day3'
# file_name = dataset2+".csv"
# data3 = pd.read_csv(file_name)

# dataset2 = 'Datasets/raw datasets day 1-6/Affordance_Day4'
# file_name = dataset2+".csv"
# data4 = pd.read_csv(file_name)

# dataset2 = 'Datasets/raw datasets day 1-6/Affordance_Day5'
# file_name = dataset2+".csv"
# data5 = pd.read_csv(file_name)

# dataset2 = 'Datasets/raw datasets day 1-6/Affordance_Day6'
# file_name = dataset2+".csv"
# data6 = pd.read_csv(file_name)



# data1_sc = data1["scenario"].drop([0,1])

# data2_sc = data2["scenario"].drop([0,1])

# data3_sc = data3["scenario"].drop([0,1])

# data4_sc = data4["scenario"].drop([0,1])

# data5_sc = data5["scenario"].drop([0,1])

# data6_sc = data6["scenario"].drop([0,1])

# frames = [data1_sc,data2_sc,data3_sc,data4_sc,data5_sc,data6_sc]

# new_data = pd.concat(frames)

combined_dataset = 'Affordance_November14_alldatacleansed_validated_new_data.csv'
file_name = 'C:/Users/sree2/Dropbox/SYR_GAship/afforadance_Study/Datasets/cleansed_data/'+combined_dataset
combo_data = pd.read_csv(file_name)



#new_data = combo_data["scenario"]#.drop([0,1])

combo_data["scenario"] = [BeautifulSoup(text,"lxml").get_text() for text in combo_data["scenario"] ]
combo_data["scenario"] = combo_data["scenario"].str.replace("\r\n","")
new_data = combo_data["scenario"].apply(lambda x: ' '.join(x.split()))


counts = pd.DataFrame(new_data.value_counts().reset_index())
counts.columns = ['scenario', 'count']




counts.to_csv("Datasets/scenario_counts/Affordance_November14_alldata_scenario_counts.csv")