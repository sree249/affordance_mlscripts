#cleanse data set

#step 1: remove the points that fail the validation tests
#step 2: remove the points that fail the attention checks

'''
Please check the dataset for headers and scenario column
'''

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup #please ensure installed

dataset = 'Affordance_November19_alldata'
file_name = "Datasets/"+dataset+".csv"
data = pd.read_csv(file_name)
print('Input data has shape:', data.shape[0])

headers = [ 'Random_ID','ResponseId','device_use','Finished','Q20', 'Q1', 'Q1_6_TEXT', 'Q2', 'Q2_4_TEXT', 'Q3', 'Q3_5_TEXT', 'Q4', 'Q4_2_TEXT', 'Q5', 'Q5_5_TEXT', 'Q6', 'Q7',
			'Q7_4_TEXT', 'Q8', 'Q8_4_TEXT', 'Q9', 'Q9_5_TEXT', 'Q10', 'Q10_5_TEXT', 'Q11', 
			'Q11_4_TEXT', 'Q12', 'Q12_4_TEXT', 'Q13', 'Q13_3_TEXT', 
			'1_Q1', '1_Q1_6_TEXT', '1_Q2', '1_Q2_4_TEXT', '1_Q3', '1_Q3_5_TEXT', '1_Q4', '1_Q4_2_TEXT', '1_Q5', 
			'1_Q5_35_TEXT', '1_Q6', '1_Q7', '1_Q8', '1_Q8_4_TEXT', '1_Q9', '1_Q9_5_TEXT', 
			'1_Q10', '1_Q10_5_TEXT', '1_Q11', '1_Q11_4_TEXT', '1_Q12', '1_Q12_4_TEXT', 
			'1_Q13', '1_Q13_3_TEXT', '2_Q1', '2_Q1_6_TEXT', '2_Q2', '2_Q2_4_TEXT', '2_Q3', '2_Q3_5_TEXT', '2_Q4', 
			'2_Q4_2_TEXT', '2_Q5', '2_Q5_35_TEXT', '2_Q6', '2_Q7', '2_Q8', '2_Q8_4_TEXT', 
			'2_Q9', '2_Q9_5_TEXT', '2_Q10', '2_Q10_5_TEXT', '2_Q11', '2_Q11_4_TEXT',  
			'2_Q12', '2_Q12_4_TEXT', '2_Q13', '2_Q13_3_TEXT', '3_Q1', '3_Q1_6_TEXT', '3_Q2', '3_Q2_4_TEXT', 
			'3_Q3', '3_Q3_5_TEXT', '3_Q4', '3_Q4_2_TEXT', '3_Q5', '3_Q5_35_TEXT', '3_Q6', '3_Q7', '3_Q8',
			'3_Q8_4_TEXT', '3_Q9', '3_Q9_5_TEXT', '3_Q10', '3_Q10_5_TEXT', '3_Q11',
			'3_Q11_4_TEXT', '3_Q12', '3_Q12_4_TEXT', '3_Q13', '3_Q13_3_TEXT', '4_Q1', 
			'4_Q1_6_TEXT', '4_Q2', '4_Q2_4_TEXT', '4_Q3', '4_Q3_5_TEXT', '4_Q4', '4_Q4_2_TEXT', '4_Q5', '4_Q5_35_TEXT',
			'4_Q6', '4_Q7', '4_Q8', '4_Q8_4_TEXT', '4_Q9', '4_Q9_5_TEXT', '4_Q10', '4_Q10_5_TEXT', 
			'4_Q11', '4_Q11_4_TEXT', '4_Q12', '4_Q12_4_TEXT', '4_Q13', '4_Q13_3_TEXT',
			'5_Q1', '5_Q1_6_TEXT', '5_Q2', '5_Q2_4_TEXT', '5_Q3', '5_Q3_5_TEXT', '5_Q4', '5_Q4_2_TEXT', '5_Q5', '5_Q5_35_TEXT',
			'5_Q6', '5_Q7', '5_Q8', '5_Q8_4_TEXT', '5_Q9', '5_Q9_5_TEXT', '5_Q10', '5_Q10_5_TEXT',
			'5_Q11', '5_Q11_4_TEXT', '5_Q12', '5_Q12_4_TEXT', '5_Q13', '5_Q13_3_TEXT', 
			'6_Q1', '6_Q1_6_TEXT', '6_Q2', '6_Q2_4_TEXT', '6_Q3', '6_Q3_5_TEXT', '6_Q4', '6_Q4_2_TEXT', '6_Q5', '6_Q5_35_TEXT',
			'6_Q6', '6_Q7', '6_Q8', '6_Q8_4_TEXT', '6_Q9', '6_Q9_5_TEXT', '6_Q10', '6_Q10_5_TEXT',
			'6_Q11', '6_Q11_4_TEXT', '6_Q12', '6_Q12_4_TEXT', '6_Q13', '6_Q13_3_TEXT','Q183','Q184','Q184_4_TEXT',
			'1_Q186','1_Q186_4_TEXT','2_Q186','2_Q186_4_TEXT','3_Q186','3_Q186_4_TEXT','4_Q186','4_Q186_4_TEXT',
			'5_Q186','5_Q186_4_TEXT','6_Q186','6_Q186_4_TEXT',
			'1_Q185','1_Q185_5_TEXT','2_Q185','2_Q185_5_TEXT','3_Q185','3_Q185_5_TEXT','4_Q185','4_Q185_5_TEXT',
			'5_Q185','5_Q185_5_TEXT','6_Q185','6_Q185_5_TEXT','scenario'] #"scenario" should be added/removed as per dataset.
'''
headers = ['ResponseId','Q6','Q183','Q11','Q184', 'Q184_4_TEXT','Q11_4_TEXT','1_Q8','1_Q186','1_Q8_4_TEXT','Q20','1_Q186_4_TEXT','1_Q10','1_Q185','1_Q10_5_TEXT','1_Q185_5_TEXT',
				'2_Q8','2_Q186','2_Q8_4_TEXT','2_Q186_4_TEXT','2_Q10','2_Q185','2_Q10_5_TEXT','2_Q185_5_TEXT',
				'3_Q8','3_Q186','3_Q8_4_TEXT','3_Q186_4_TEXT','3_Q10','3_Q185','3_Q10_5_TEXT','3_Q185_5_TEXT',
				'4_Q8','4_Q186','4_Q8_4_TEXT','4_Q186_4_TEXT','4_Q10','4_Q185','4_Q10_5_TEXT','4_Q185_5_TEXT',
				'5_Q8','5_Q186','5_Q8_4_TEXT','5_Q186_4_TEXT','5_Q10','5_Q185','5_Q10_5_TEXT','5_Q185_5_TEXT',
				'6_Q8','6_Q186','6_Q8_4_TEXT','6_Q186_4_TEXT','6_Q10','6_Q185','6_Q10_5_TEXT','6_Q185_5_TEXT']
'''
res_data = data[headers].drop([0,1]).dropna(subset=['Q20']).fillna("no_values")

''''
HAD TO MODIFY FOR CURRENT DATASET ONLY ####### do REMOVE:
use only if HEADERS ARE ABSENT
'''
#res_data = data[headers].dropna(subset=['Q20']).fillna("no_values")




#print(res_data)
file_name = dataset+"_cleansed.csv"
res_data.to_csv(file_name)

'''
res_data['a_checks'] = res_data.apply(lambda x : 1 if (x['Q6'].equals(x['Q183'])) & (x['Q11'].equals(x['Q184'])) & 
									(x['1_Q8'].equals(x['1_Q186'])) & (x['3_Q8'].equals(x['3_Q186']))  
										& (x['4_Q8'].equals(x['4_Q186'])) & (x['5_Q8'].equals(x['5_Q186'])) & (x['6_Q8'].equals(x['6_Q186'])) &
										(x['1_Q10'].equals(x['1_Q185'])) & (x['2_Q10'].equals(x['2_Q185'])) & (x['3_Q10'].equals(x['3_Q185'])) & 
										(x['4_Q10'].equals(x['4_Q185'])) & (x['5_Q10'].equals(x['5_Q185'])) & (x['6_Q10'].equals(x['6_Q185']))
										else 0, axis=1)
'''
'''
res_data['a_checks_1'] = res_data.apply(lambda x : 1 if (x['Q6'] == (x['Q183'])) else 0,axis=1)
res_data['a_checks_2'] = res_data.apply(lambda x : 1 if (x['Q11'] == (x['Q184'])) else 0,axis=1)	
res_data['a_checks_3'] = res_data.apply(lambda x : 1 if (x['Q184_4_TEXT'] == (x['Q11_4_TEXT'])) else 0,axis=1)
#print(res_data['Q11_4_TEXT'])


res_data['a_checks_4'] = res_data.apply(lambda x : 1 if (x['1_Q8'] == (x['1_Q186'])) else 0,axis=1)
res_data['a_checks_5'] = res_data.apply(lambda x : 1 if (x['1_Q8_4_TEXT'] == (x['1_Q186_4_TEXT'])) else 0,axis=1)
res_data['a_checks_6'] = res_data.apply(lambda x : 1 if (x['1_Q10'] == (x['1_Q185'])) else 0,axis=1)
res_data['a_checks_7'] = res_data.apply(lambda x : 1 if (x['1_Q10_5_TEXT'] == (x['1_Q185_5_TEXT'])) else 0,axis=1)


res_data['a_checks_8'] = res_data.apply(lambda x : 1 if (x['2_Q8'] == (x['2_Q186'])) else 0,axis=1)
res_data['a_checks_9'] = res_data.apply(lambda x : 1 if (x['2_Q8_4_TEXT'] == (x['2_Q186_4_TEXT'])) else 0,axis=1)
res_data['a_checks_10'] = res_data.apply(lambda x : 1 if (x['2_Q10'] == (x['2_Q185'])) else 0,axis=1)
res_data['a_checks_11'] = res_data.apply(lambda x : 1 if (x['2_Q10_5_TEXT'] == (x['2_Q185_5_TEXT'])) else 0,axis=1)
res_data['a_checks_12'] = res_data.apply(lambda x : 1 if (x['3_Q8'] == (x['3_Q186'])) else 0,axis=1)
res_data['a_checks_13'] = res_data.apply(lambda x : 1 if (x['3_Q8_4_TEXT'] == (x['3_Q186_4_TEXT'])) else 0,axis=1)
res_data['a_checks_14'] = res_data.apply(lambda x : 1 if (x['3_Q10'] == (x['3_Q185'])) else 0,axis=1)
res_data['a_checks_15'] = res_data.apply(lambda x : 1 if (x['3_Q10_5_TEXT'] == (x['3_Q185_5_TEXT'])) else 0,axis=1)
res_data['a_checks_16'] = res_data.apply(lambda x : 1 if (x['4_Q8'] == (x['4_Q186'])) else 0,axis=1)
res_data['a_checks_17'] = res_data.apply(lambda x : 1 if (x['4_Q8_4_TEXT'] == (x['4_Q186_4_TEXT'])) else 0,axis=1)
res_data['a_checks_18'] = res_data.apply(lambda x : 1 if (x['4_Q10'] == (x['4_Q185'])) else 0,axis=1)
res_data['a_checks_19'] = res_data.apply(lambda x : 1 if (x['4_Q10_5_TEXT'] == (x['4_Q185_5_TEXT'])) else 0,axis=1)
res_data['a_checks_20'] = res_data.apply(lambda x : 1 if (x['5_Q8'] == (x['5_Q186'])) else 0,axis=1)
res_data['a_checks_21'] = res_data.apply(lambda x : 1 if (x['5_Q8_4_TEXT'] == (x['5_Q186_4_TEXT'])) else 0,axis=1)
res_data['a_checks_22'] = res_data.apply(lambda x : 1 if (x['5_Q10'] == (x['5_Q185'])) else 0,axis=1)
res_data['a_checks_23'] = res_data.apply(lambda x : 1 if (x['5_Q10_5_TEXT'] == (x['5_Q185_5_TEXT'])) else 0,axis=1)
res_data['a_checks_24'] = res_data.apply(lambda x : 1 if (x['6_Q8'] == (x['6_Q186'])) else 0,axis=1)
res_data['a_checks_25'] = res_data.apply(lambda x : 1 if (x['6_Q8_4_TEXT'] == (x['6_Q186_4_TEXT'])) else 0,axis=1)
res_data['a_checks_26'] = res_data.apply(lambda x : 1 if (x['6_Q10'] == (x['6_Q185'])) else 0,axis=1)
res_data['a_checks_27'] = res_data.apply(lambda x : 1 if (x['6_Q10_5_TEXT'] == (x['6_Q185_5_TEXT'])) else 0,axis=1)
'''

	
							
										
#res_data['a_checks_Q6'] = res_data.apply(lambda res_data : 1 if (res_data['Q6'] == res_data['Q183'])and (res_data['Q11'] == res_data['Q184']) and
#(res_data['1_Q8'].str.equals(res_data['1_Q186'])) else 0, axis=1)


m = (res_data['Q6']==(res_data['Q183'])) & (res_data['Q11'] == (res_data['Q184'])) & (res_data['Q184_4_TEXT'] == (res_data['Q11_4_TEXT']))\
& (res_data['1_Q8'] == (res_data['1_Q186'])) & (res_data['2_Q8'] == (res_data['2_Q186'])) \
& (res_data['1_Q8_4_TEXT'] == (res_data['1_Q186_4_TEXT'])) & (res_data['2_Q8_4_TEXT'] == (res_data['2_Q186_4_TEXT']))\
& (res_data['3_Q8'] == (res_data['3_Q186'])) & (res_data['4_Q8'] == (res_data['4_Q186'])) \
& (res_data['3_Q8_4_TEXT'] == (res_data['3_Q186_4_TEXT'])) & (res_data['4_Q8_4_TEXT'] == (res_data['4_Q186_4_TEXT']))\
& (res_data['5_Q8'] == (res_data['5_Q186'])) & (res_data['6_Q8'] == (res_data['6_Q186'])) \
& (res_data['5_Q8_4_TEXT'] == (res_data['5_Q186_4_TEXT'])) & (res_data['6_Q8_4_TEXT'] == (res_data['6_Q186_4_TEXT'])) \
& (res_data['1_Q10'] == (res_data['1_Q185'])) & (res_data['2_Q10']==(res_data['2_Q185'])) \
& (res_data['1_Q10_5_TEXT'] == (res_data['1_Q185_5_TEXT'])) & (res_data['2_Q10_5_TEXT'] == (res_data['2_Q185_5_TEXT']))\
& (res_data['3_Q10'] == (res_data['3_Q185'])) & (res_data['4_Q10'] == (res_data['4_Q185'])) \
& (res_data['3_Q10_5_TEXT'] == (res_data['3_Q185_5_TEXT'])) & (res_data['4_Q10_5_TEXT'] == (res_data['4_Q185_5_TEXT']))\
& (res_data['5_Q10'] == (res_data['5_Q185'])) & (res_data['6_Q10'] == (res_data['6_Q185'])) \
& (res_data['5_Q10_5_TEXT'] == (res_data['5_Q185_5_TEXT'])) & (res_data['6_Q10_5_TEXT'] == (res_data['6_Q185_5_TEXT']))
res_data['a_checks'] = np.where(m,1,0)

#preprocessing for scenario data
res_data["scenario"] = [BeautifulSoup(text,"lxml").get_text() for text in res_data["scenario"] ]
res_data["scenario"] = res_data["scenario"].str.replace("\r\n","")
res_data["scenario"] = res_data["scenario"].apply(lambda x: ' '.join(x.split()))








#validate a_checks
check_heads = ['Random_ID','ResponseId','Q6','Q183','Q11','Q184', 'Q184_4_TEXT','Q11_4_TEXT','1_Q8','1_Q186','1_Q8_4_TEXT','Q20','1_Q186_4_TEXT','1_Q10','1_Q185','1_Q10_5_TEXT','1_Q185_5_TEXT',
				'2_Q8','2_Q186','2_Q8_4_TEXT','2_Q186_4_TEXT','2_Q10','2_Q185','2_Q10_5_TEXT','2_Q185_5_TEXT',
				'3_Q8','3_Q186','3_Q8_4_TEXT','3_Q186_4_TEXT','3_Q10','3_Q185','3_Q10_5_TEXT','3_Q185_5_TEXT',
				'4_Q8','4_Q186','4_Q8_4_TEXT','4_Q186_4_TEXT','4_Q10','4_Q185','4_Q10_5_TEXT','4_Q185_5_TEXT',
				'5_Q8','5_Q186','5_Q8_4_TEXT','5_Q186_4_TEXT','5_Q10','5_Q185','5_Q10_5_TEXT','5_Q185_5_TEXT',
				'6_Q8','6_Q186','6_Q8_4_TEXT','6_Q186_4_TEXT','6_Q10','6_Q185','6_Q10_5_TEXT','6_Q185_5_TEXT']
check_heads.append('a_checks')			
check_data = res_data[check_heads]

file_name = "Validations/"+dataset+"_attentionchecks.csv"
check_data.to_csv(file_name)	

#write to attention_checks
file_name = "Validations/attention_checks/"+dataset+"_withattentonchecks.csv"
res_data.to_csv(file_name)

res_data = res_data[res_data['a_checks']==1]
#rewrite the dataset after removing attention checks
file_name = "Datasets/cleansed_data/"+dataset+"cleansed_validated_new_data.csv"			
res_data.to_csv(file_name)


									
					

