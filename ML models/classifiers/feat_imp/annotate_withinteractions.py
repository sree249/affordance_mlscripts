#run script for annotation of feature importance with main and interaction terms

import pandas as pd
import numpy as np

file_name = 'Annotated_feature_imp_Affordance_November4_alldata_tree_Laptop.csv'
data = pd.read_csv(file_name,encoding = "ISO-8859-1")

res_data = data[['features','Question','Answer option']]

device_map = ['Laptop','Smart_Phone','Desktop_Computer','Tablet','Smart_Speaker','Smart_Watch']
device = device_map[5] 
file_to_annotate = 'feature_imp_Affordance_November4_alldata_selectkbest_interactions'+device+'.csv'



to_annotate = pd.read_csv(file_to_annotate,encoding = "ISO-8859-1",index_col=0)


res_data["main_features"] = res_data["features"].str.replace("Laptop", device)

res_data["Question"] = res_data["Question"].str.replace("Laptop", device)

to_annotate["scen_feat"] = to_annotate["features"].str.split(":").str[0]

to_annotate["main_features"] = to_annotate["features"].str.split(":").str[1]

res_data = res_data.drop("features",axis=1)
#merge data
final_data = pd.merge(to_annotate,res_data,on="main_features")
final_data["Affordance_Question"] = final_data["Question"]
final_data["Affordance_Answer"] = final_data["Answer option"]

final_data = final_data.drop('Question',axis=1)
final_data = final_data.drop('Answer option',axis=1)

res_data["scen_feat"] = res_data["main_features"]
res_data = res_data.drop("main_features",axis=1)

final_data = pd.merge(final_data,res_data,on="scen_feat")

final_data["scen_Question"] = final_data["Question"]
final_data["scen_Answer"] = final_data["Answer option"]

final_data = final_data.drop('Question',axis=1)
final_data = final_data.drop('Answer option',axis=1)

final_data = final_data.sort_values(["Importance"],ascending=False)

fin_file_name = "Annotated_"+file_to_annotate
final_data.to_csv(fin_file_name)






