# test encoding with numeric values
#Using pandas is more efficient for larger datasets hence switching to pandas.
import pandas as pd
from bs4 import BeautifulSoup

dataset = 'Affordance_November19_alldata'
file_name = 'Datasets/cleansed_data/'+dataset+"cleansed_validated_new_data.csv"
data = pd.read_csv(file_name)

#code to get the headers:
#print(list(data))

#initialize the headers:
headers = [ 'ResponseId','device_use','Q20', 'Q1', 'Q1_6_TEXT', 'Q2', 'Q2_4_TEXT', 'Q3', 'Q3_5_TEXT', 'Q4', 'Q4_2_TEXT', 'Q5', 'Q5_5_TEXT', 'Q6', 'Q7',
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
			'6_Q11', '6_Q11_4_TEXT', '6_Q12', '6_Q12_4_TEXT', '6_Q13', '6_Q13_3_TEXT','scenario']
			
#res_data = data[headers].drop([0,1]).dropna(subset=['Q20']).fillna("no_values")

res_data = data[headers]

res_data = res_data.applymap(str)
#res_data['deviceCount'] = res_data['Q20'].str.split(",").str.len()

#res_data = res_data[res_data['deviceCount'] >=3]



#using the numeric codes of the options -> this way we can avoid string matching errors
range_questions = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13'] 
no_questions = {'Q1':7,'Q2':5,'Q3':6,'Q4':3,'Q5':6,'Q6':4,'Q7':5,'Q8':5,'Q9':6,'Q10':7,'Q11':5,'Q12':5,'Q13':4}
#create a map from each question to the individual sets for string -> dynamic variable access
question_map = {}
question_feat_map = {} #mapping the questions directly to the feature labels ### to be used in validation
feat_map = {} # create a feature map from the numeric options to features (avoid collisions by ques+"_"+option no. as key

for ques in range_questions:
	#generate option set for each question
	# NOTE Q6: does not have others option so it is to be different
	#NOTE Q10: has coding with others is 2 before the last option needs separate treatment
	if ques == 'Q6':
		question_map[ques] = [str(x+1) for x in range(0,no_questions[ques])]
	elif ques == 'Q10':
		question_map[ques] = [str(x+1) for x in range(0,no_questions[ques]-3)] # not counting the text entry field.
		question_map[ques].append(str(no_questions[ques]-1)) #adjust for text entry
		question_map[ques].append(str(no_questions[ques])) # last response code
	else:
		question_map[ques] = [str(x+1) for x in range(0,no_questions[ques]-2)] # not counting the text entry field.
		question_map[ques].append(str(no_questions[ques])) # last response code
		
	

#print(question_map)
	
#connect questions to features
for ques in question_map.keys(): # for each of the headers	
		cur_set = question_map[ques] # get the set of questions
		cur_labels = []
		for e in cur_set:
			feat_map[ques+"_"+e] = ques+"_feat"+e
			cur_labels.append(ques+"_feat"+e)
		question_feat_map[ques]= cur_labels
		
		
#print(feat_map)
#print(question_feat_map)

#add the binary encodings (i.e feature reps. for each question) 
for ques in question_map.keys(): # for each of the headers	
		cur_set = question_map[ques] # get the set of questions
		for e in cur_set:
			#add a col for each feature from the current question set
			feat_key = ques+"_"+e
			res_data[feat_map[feat_key]] = (res_data[ques].str.contains(e)).astype(int)
			
			

#forming the header for the parsed files for PART 1
head = ['ResponseId']
for qs in range_questions:
		head.append(qs)
		head.extend(question_feat_map[qs])

#VALIDATION TESTS TO BE REMOVED LATER	
check_data = res_data[head]
file_name = "Validations/"+dataset+"_part1_encoding.csv"
check_data.to_csv(file_name)

######################################################################################################################

# Device use and part 2

#create a mapping from the choice to the numeric data
#order of both 'Device Use' and 'Q20' codes are the same so we can use one map 
device_map = {'1':'Laptop', '2':'Smart_Phone', '3':'Desktop_Computer', '4':'Tablet', '5':'Smart_Speaker', '6':'Smart_Watch'}

#break the data device wise
laptop_data = res_data[res_data['device_use'].str.contains('1')]
laptop_data['actual_use'] = (res_data['Q20'].str.contains('1')).astype(int)

smartphone_data = res_data[res_data['device_use'].str.contains('2')]
smartphone_data['actual_use'] = (res_data['Q20'].str.contains('2')).astype(int)

desktop_data = res_data[res_data['device_use'].str.contains('3')]
desktop_data['actual_use'] = (res_data['Q20'].str.contains('3')).astype(int)

tablet_data = res_data[res_data['device_use'].str.contains('4')]
tablet_data['actual_use'] = (res_data['Q20'].str.contains('4')).astype(int)

smartspeaker_data = res_data[res_data['device_use'].str.contains('5')]
smartspeaker_data['actual_use'] = (res_data['Q20'].str.contains('5')).astype(int)

smartwatch_data = res_data[res_data['device_use'].str.contains('6')]
smartwatch_data['actual_use'] = (res_data['Q20'].str.contains('6')).astype(int)


#some tests for validation

# laptop_data.to_csv("Validations/"+dataset+"_Laptop_data.csv")
# smartphone_data.to_csv("Validations/smartphone_data.csv")
# desktop_data.to_csv("Validations/Desktop_data.csv")
# tablet_data.to_csv("Validations/tablet_data.csv")
# smartspeaker_data.to_csv("Validations/smartspeaker_data.csv")
# smartwatch_data.to_csv("Validations/smartWatch_data.csv")


device_data_map = {'1':laptop_data, '2':smartphone_data, '3':desktop_data, '4':tablet_data, '5':smartspeaker_data, '6':smartwatch_data}

#reuse question map as the number of questions in part 2 is the same, reused for each device


### FOLLOWING AT DEVICE LEVEL
#add the binary encodings (i.e feature reps. for each question) 
for dev in device_data_map.keys():
	cur_data = device_data_map[dev]
	
	for ques in question_map.keys(): # for each of the headers	
		cur_set = question_map[ques] # get the set of questions
		cur_dev_ques = dev+"_"+ques
		for e in cur_set:
			#add a col for each feature from the current question set
			new_key = device_map[dev]+'_'+ques+"_feat"+e		
			cur_data[new_key] = (cur_data[cur_dev_ques].str.contains(e)).astype(int)
			

	head = []
	for qs in range_questions:
		head.extend(question_feat_map[qs])
			

	device_head = ['ResponseId','device_use','Q20','actual_use']
	for qs in range_questions:
		device_head.append(dev+'_'+qs)
		for ele in question_feat_map[qs]:
			cur_feat = device_map[dev]+'_'+ele
			device_head.append(cur_feat)

	check_device_data = cur_data[device_head]
	file_name = "Validations/Validate_part2_"+device_map[dev]+".csv"
	check_device_data.to_csv(file_name)

	device_head = []
	for qs in range_questions:
		for ele in question_feat_map[qs]:
			cur_feat = device_map[dev]+'_'+ele
			device_head.append(cur_feat)
			

			
	fin_header = ['ResponseId']
	fin_header.extend(head) # add part 1
	fin_header.extend(device_head)
	fin_header.append('device_use')
	fin_header.append('Q20')
	fin_header.append('actual_use')
	fin_header.append('scenario')
	fin_cur_data = cur_data[fin_header]
	
	#adding scenario driven feature
	fin_cur_data["scenario"] = [BeautifulSoup(text,"lxml").get_text() for text in fin_cur_data["scenario"] ]
	fin_cur_data["scenario"] = fin_cur_data["scenario"].str.replace("\r\n","")
	fin_cur_data["scenario"] = fin_cur_data["scenario"].apply(lambda x: ' '.join(x.split()))
	fin_cur_data["raw_scenario"] = fin_cur_data["scenario"].str.replace(" ","") #remove spaces for string matching
	
	#merge with the scenario encoding file to find the features from the scenarios
	scenarios_data = pd.read_csv("Datasets/scenario_category_cleansed.csv",encoding = "ISO-8859-1",index_col=0)
	final_data = pd.merge(fin_cur_data,scenarios_data,on="raw_scenario")
	
	file_name = "Datasets/Encoded_files/"+"encoded_"+dataset+"_"+device_map[dev]+"_data.csv"
	#fin_cur_data.to_csv(file_name)
	final_data.to_csv(file_name)		

 












		

			
	

