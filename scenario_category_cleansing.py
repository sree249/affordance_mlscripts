#scenario propreprocessing

#trying pandas dataset concacts

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup



combined_dataset = 'scenario_category.csv'
file_name = 'C:/Users/sree2/Dropbox/SYR_GAship/afforadance_Study/Datasets/'+"scenario_category.csv"
combo_data = pd.read_csv(file_name)

combo_data["scenario"] = [BeautifulSoup(text,"lxml").get_text() for text in combo_data["scenario"] ]
combo_data["scenario"] = combo_data["scenario"].str.replace("\r\n","")
combo_data["raw_scenario"] = combo_data["scenario"].str.replace(" ","")
combo_data["org_scenaio"] = combo_data["scenario"]
combo_data = combo_data.drop("scenario",axis=1)
combo_data_imd = pd.get_dummies(combo_data,columns=["Location","Relationship"],prefix=['location','Relationship'])
combo_data = pd.concat([combo_data, combo_data_imd], axis=1) 
combo_data = combo_data.iloc[:,~combo_data.columns.duplicated()]


#combo_data = combo_data.join(one_hot)
#combo_data = combo_data.drop("Location",axis=1)





output_file = "Datasets/scenario_category_cleansed.csv"

combo_data.to_csv(output_file)