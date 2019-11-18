#run script for annotation of feature importance

import pandas as pd
import numpy as np

file_name = 'Annotated_feature_imp_Affordance_November4_alldata_tree_Laptop.csv'
data = pd.read_csv(file_name,encoding = "ISO-8859-1")

res_data = data[['features','Question','Answer option']]

device_map = ['Smart Phone','Desktop Computer','Tablet','Smart Speaker','Smart Watch']
device = device_map[4] 
file_to_annotate = 'feature_imp_Affordance_November4_alldata_tree_'+device+'.csv'



to_annotate = pd.read_csv(file_to_annotate,encoding = "ISO-8859-1",index_col=0)


res_data["features"] = res_data["features"].str.replace("Laptop", device)

res_data["Question"] = res_data["Question"].str.replace("Laptop", device)

#merge data
final_data = pd.merge(to_annotate,res_data,on="features")

fin_file_name = "Annotated_"+file_to_annotate
final_data.to_csv(fin_file_name)






