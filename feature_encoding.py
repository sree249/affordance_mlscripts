#Using pandas is more efficient for larger datasets hence switching to pandas.
import pandas as pd

data = pd.read_csv('test_data.csv')

#code to get the headers:
#print(list(data))

#initialize the headers:
headers = [ 'ResponseId','Q1', 'Q1_6_TEXT', 'Q2', 'Q2_4_TEXT', 'Q3', 'Q3_5_TEXT', 'Q4', 'Q4_2_TEXT', 'Q5', 'Q5_5_TEXT', 'Q6', 'Q7',
			'Q7_4_TEXT', 'Q8', 'Q8_4_TEXT', 'Q9', 'Q9_5_TEXT', 'Q10', 'Q10_5_TEXT', 'Q11', 
			'Q11_4_TEXT', 'Q12', 'Q12_4_TEXT', 'Q13', 'Q13_3_TEXT']
			


			
res_data = data[headers].drop([0,1]).dropna(subset=headers[1:],how='all').fillna("no_values")



#print(res_data)

#list of question markers
# *** NOTE ***: When feeding in () can cause issues with the regex so replace the part with *
#Q1:

Q1_set = ['Voice-based interaction','Touch-based interaction e.g, through scrolling, tapping','Text-based interaction with a digital pen/Stylus',
		  'Text-based interaction using physical keyboard','Text-based interaction using virtual/digital keyboard on a screen',
		  'The input modality does not matter to me']
		  
#Q2

Q2_set = ['Use/view another application through touch-based input', 'Use/view another application through voice-based input', 
		  'The modality of input text- or voice-based does not matter as long as I am able to use another application',
		  'I would not like to multitask while performing this task']
		  
#Q3

Q3_set = ['It should be easy to find and understand the relevant features of the devices to complete the given task.', 
		  'The device should provide few and simple steps to complete the task.',
		  'The devices should be able to display any task related information e.g, response on a screen',
		  'The devices should be able to provide any task related information as an audio response.',
		  'Ease of use does not hold any relevance for me with respect to this task']
		  
		  
#Q4

Q4_set = ['Being able to move around while performing the task','Mobility of the device does not matter to me in this task']

#Q5

Q5_set = ['It should be able to execute the task without any errors','The device can make errors sometimes, but should execute the task perfectly most of the time',
		  'The device can make some errors, but it should correct itself by learning from my previous actions or more interactions with me.',
		  

		  
		  
		  
#TO DO: add in loops for all maps from questions in the header
range_questions = ['Q1','Q2','Q3'] #finally range question should be header
#create a map from each question to the individual sets for string -> dynamic variable access
question_map = {}
question_feat_map = {} #mapping the questions directly to the feature labels ### to be used in validation
for e in range_questions:
	if e == 'Q1':
		question_map[e] = Q1_set 
	elif e == 'Q2':
		question_map[e] = Q2_set
	else: 
		question_map[e] = Q3_set
		

#FINAL feature headers map
feat_map = {}

#connect questions to features
for ques in question_map.keys(): # for each of the headers	
		cur_set = question_map[ques] # get the set of questions
		cur_labels = []
		for ind in range(0,len(cur_set)):
			feat_map[cur_set[ind]]=ques+"_feat_"+str(ind+1)
			cur_labels.append(ques+"_feat_"+str(ind+1))
		question_feat_map[ques] = cur_labels
	

#print(feat_map)
#print(question_feat_map)

#add the binary encodings (i.e feature reps. for each question) 
for ques in question_map.keys(): # for each of the headers	
		cur_set = question_map[ques] # get the set of questions
		for e in cur_set:
		#add a col for each feature from the current question set
			res_data[ques] = res_data[ques].str.replace('(', '')
			res_data[ques] = res_data[ques].str.replace(')', '')
			res_data[feat_map[e]] = (res_data[ques].str.contains(e)).astype(int)
	
	
#VALIDATION TESTS TO BE REMOVED LATER
head = []
for qs in range_questions:
		head.append(qs)
		head.extend(question_feat_map[qs])

# print(head)
check_data = res_data[head]
check_data.to_csv("Validate_part1_encoding.csv")

# #drop the column from the dataframe
# res_data.drop('Q1',axis=1,inplace=True)

# print(res_data[['ResponseId','Q1_feat_1','Q1_feat_2']])

