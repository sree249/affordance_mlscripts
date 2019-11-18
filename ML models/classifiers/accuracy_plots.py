# Basic ML classifiers

#imports
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from  sklearn.ensemble import GradientBoostingClassifier
from  sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

#resample
	
from sklearn.utils import resample


#import data 
datapath = 'Datasets/Encoded_files/'
filename = 'encoded_Affordance_November19_alldata'
#device_list = ['Laptop', 'Smart Phone', 'Desktop Computer','Tablet','Smart Speaker','Smart Watch']

device_list = ['Laptop','Smart_Phone','Desktop_Computer','Smart_Watch','Tablet','Smart_Speaker']
classifier_list = ['Decisiontree', 'SVM','RandomForest','Adaboost','logistic']

classifier_index = 0
classifier = classifier_list[classifier_index]

reduction_list = ['tree','variancethreshold','selectkbest','pca','none']
reduction_list_ind = 2
reduction = reduction_list[reduction_list_ind]

Data_models = ['part1', 'part1_loc','part1_rel','part1_loc_rel','part2','part2_loc','part2_rel','part2_loc_rel','part1_part2',
				'part1_part2_loc','part1_part2_rel','part1_part2_loc_rel']
data_model_ind = 8
retrieve_features = True


head = [ 'Q1_feat1', 'Q1_feat2', 'Q1_feat3', 'Q1_feat4', 'Q1_feat5', 'Q1_feat7', 'Q2_feat1', 'Q2_feat2', 'Q2_feat3', 'Q2_feat5', 'Q3_feat1', 'Q3_feat2', 
			'Q3_feat3', 'Q3_feat4', 'Q3_feat6', 'Q4_feat1', 'Q4_feat3', 'Q5_feat1', 'Q5_feat2', 'Q5_feat3', 'Q5_feat4', 'Q5_feat6', 'Q6_feat1', 'Q6_feat2',
			'Q6_feat3', 'Q6_feat4', 'Q7_feat1', 'Q7_feat2', 'Q7_feat3', 'Q7_feat5', 'Q8_feat1', 'Q8_feat2', 'Q8_feat3', 'Q8_feat5', 'Q9_feat1', 'Q9_feat2',
			'Q9_feat3', 'Q9_feat4', 'Q9_feat6', 'Q10_feat1', 'Q10_feat2', 'Q10_feat3', 'Q10_feat4', 'Q10_feat6', 'Q10_feat7', 'Q11_feat1', 'Q11_feat2', 
			'Q11_feat3', 'Q11_feat5', 'Q12_feat1', 'Q12_feat2', 'Q12_feat3', 'Q12_feat5', 'Q13_feat1', 'Q13_feat2', 'Q13_feat4' ]
			


for device in device_list:
	print("##########################################")
	print("current device dataset:", device)
	file = 'C:/Users/sree2/Dropbox/SYR_GAship/afforadance_Study/Datasets/Encoded_files/encoded_Affordance_November19_alldata_'+device+'_data.csv'
	data = pd.read_csv(file)

	
	training_head = []
	#training_head.extend(head)
	dev_heads = []
	for e in head:
		cur = device+"_"+e
		dev_heads.append(cur)
		
	
	#adding in scenario_variables:
	scenario_heads = ['location_1','location_2','location_3','Relationship_1','Relationship_2','Relationship_3']
	
	#training_head = head
	#training_head.extend(scenario_heads)
	cur_model = Data_models[data_model_ind]
	if 'part1' in cur_model:
		training_head.extend(head)
	if 'part2' in cur_model:
		training_head.extend(dev_heads)
	if 'loc' in cur_model:
		scen_vars = scenario_heads[:3]
		training_head.extend(scen_vars)
	if 'rel' in cur_model:
		scen_vars = scenario_heads[3:]
		training_head.extend(scen_vars)

	
	print('the current model:',cur_model)
	print('the number of features in cur model', len(training_head))
	
	
	data_zeros = data[data['actual_use'] ==0]
	data_ones = data[data['actual_use'] ==1]
	
	one_values = data['actual_use'].value_counts()[1]
	
	zero_values = data['actual_use'].value_counts()[0]
	
	data_resample = data_zeros
	data_points = one_values
	alt_data = data_ones
	
	if zero_values <= one_values:
		data_resample = data_ones
		data_points = zero_values
		alt_data = data_zeros
	
	data_downsampled = resample(data_resample, 
											replace=False,    # sample without replacement
											n_samples=data_points,     # to match minority class
											random_state=123) # reproducible results

	data = pd.concat([data_downsampled, alt_data])
			
	

	x_data = data[training_head]

	x_data = x_data.values.astype(float)
	
	

	y_data = data["actual_use"]
	
	
	y_data = y_data.values.astype(float)

	skf = StratifiedKFold(n_splits=10)
	acc_list = [] # a list to collect all the accuracies for a particular device and plot
	f1_score_list = []
	num_feat_list = [5,10,15,20,25,30,35,40,45,50]
	num_feat_dict = {}
	
	clf1 = []
	if classifier == "Decisiontree":
		clf1 = tree.DecisionTreeClassifier()
	elif classifier == "SVM" :
		clf1 = svm.SVC(kernel='rbf', C=10,probability=True)
	elif classifier == "RandomForest":
		clf1 = RandomForestClassifier()
	elif classifier == "Adaboost":
		clf1 = AdaBoostClassifier()
	else:
		clf1 = LogisticRegression()
	
	transform = SelectKBest(chi2)

	clf = Pipeline([('anova', transform), ('classifier', clf1)])
	
	for num_feat in num_feat_list:
		clf.set_params(anova__k=num_feat)

		#this_scores = cross_val_score(clf, x_data, y_data, cv=10, n_jobs=1)
		#acc_list.append(this_scores.mean())
		
		y_pred = cross_val_predict(clf, x_data, y_data, cv=skf)
		# pred_score_f1 = f1_score(y_data, y_pred)
		score = accuracy_score(y_data,y_pred)
		
		# print("")
		# print("num of features:", num_feat)
		# print('accuracy score', score)
		# print("F1 score:",pred_score_f1)
	
		#append to accuracy list
		acc_list.append(score)
		# num_feat_dict[num_feat] = score
		# f1_score_list.append(pred_score_f1)
		
		# accuracy_frame = pd.DataFrame(list(num_feat_dict.items()),columns=['num_features','Accuracy_values'])
		# accuracy_frame['F1_values'] = f1_score_list
		#accuracy_frame = feature_frame_selected.sort_values(["Accuracy_values"],ascending=False)
		#feat_file = 'variations_num_features/metric_Scores'+filename+"_"+cur_model+"_"+reduction+"_"+device+".csv"
		#accuracy_frame.to_csv(feat_file)
		
		
		
	
	
	##########33 PLOTTING OF VALUES #####################################################333333333
	f = plt.figure()
	plt.plot(num_feat_list,acc_list,'ob--', color='black')
	plt.yticks(np.arange(min(acc_list)-0.05, max(acc_list)+0.05, 0.01))
	title = "Accuracy vs number of features for "+device
	plt.suptitle(title)
	#plt.ylabel("Accuracy")
	plt.xlabel("Number of features")
	plt.show()
	file_name = "variations_num_features/cur_plots_accuracies_withcross_val_new_v1"+reduction+"_"+device+".pdf"
	f.savefig(file_name, bbox_inches='tight')
	
	# f = plt.figure()
	# plt.plot(num_feat_list,f1_score_list,'ob--', color='black')
	# #plt.yticks(np.arange(min(f1_score_list)-0.005, max(f1_score_list)+0.005, 0.05))
	# title = "F1 scores vs number of features for "+device
	# plt.suptitle(title)
	# #plt.ylabel("F1 scores")
	# plt.xlabel("Number of features")
	# plt.show()
	# file_name = "variations_num_features/cur_plots_f1scores_"+reduction+"_"+device+".pdf"
	# f.savefig(file_name, bbox_inches='tight')

	
	

	