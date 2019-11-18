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
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC

#resample
	
from sklearn.utils import resample

#set seed at the top, to ensure uniform sampling
np.random.seed(12345)

def participant_analysis(big_table,tmp_list):
	
	means = []
	f1_score = []
	for i in tmp_list:
		tmp_value = big_table['actual_use_'+i] - big_table['Predictions_'+i]
		tmp_val_add = big_table['actual_use_'+i] + big_table['Predictions_'+i]
		tmp_abs = tmp_value.abs()
		means.append(tmp_abs.mean())
		tp = tmp_val_add[tmp_val_add > 1]
		tp = len(tp.tolist())
		tn = tmp_val_add[tmp_val_add < 1]
		tn = len(tn.tolist())
		fn = tmp_value[tmp_value > 0]
		fn = len(fn.tolist())
		fp = tmp_value[tmp_value < 0].abs()
		fp = len(fp.tolist())
		
		precision = tp/(tp+fn)
		recall = tp/ (tp+fp)
		
		f1 = 2.0*precision*recall/ (precision+recall)
		f1_score.append(f1)
	final_value = 1-np.mean(means)
	final_f1 = np.mean(f1_score)
	print(final_f1)
	return final_value



#import data 
datapath = 'Datasets/Encoded_files/'
filename = 'encoded_Affordance_November19_alldata'
#device_list = ['Laptop', 'Smart Phone', 'Desktop Computer','Tablet','Smart Speaker','Smart Watch']

device_list = ['Laptop','Smart_Phone','Desktop_Computer','Smart_Watch','Tablet','Smart_Speaker']
classifier_list = ['Decisiontree', 'SVM','RandomForest','Adaboost','logistic']

classifier_index = 1
classifier = classifier_list[classifier_index]

Data_models = ['part1', 'part1_loc','part1_rel','part1_loc_rel','part2','part2_loc','part2_rel','part2_loc_rel','part1_part2',
				'part1_part2_loc','part1_part2_rel','part1_part2_loc_rel']
data_model_ind = 11
retrieve_features = True

'''
#head = [ 'Q1_feat1', 'Q1_feat2', 'Q1_feat3', 'Q1_feat4', 'Q1_feat5', 'Q1_feat7', 'Q2_feat1', 'Q2_feat2', 'Q2_feat3', 'Q2_feat5', 'Q3_feat1', 'Q3_feat2', 
			'Q3_feat3', 'Q3_feat4', 'Q3_feat6', 'Q4_feat1', 'Q4_feat3', 'Q5_feat1', 'Q5_feat2', 'Q5_feat3', 'Q5_feat4', 'Q5_feat6', 'Q6_feat1', 'Q6_feat2',
			'Q6_feat3', 'Q6_feat4', 'Q7_feat1', 'Q7_feat2', 'Q7_feat3', 'Q7_feat5', 'Q8_feat1', 'Q8_feat2', 'Q8_feat3', 'Q8_feat5', 'Q9_feat1', 'Q9_feat2',
			'Q9_feat3', 'Q9_feat4', 'Q9_feat6', 'Q10_feat1', 'Q10_feat2', 'Q10_feat3', 'Q10_feat4', 'Q10_feat6', 'Q10_feat7', 'Q11_feat1', 'Q11_feat2', 
			'Q11_feat3', 'Q11_feat5', 'Q12_feat1', 'Q12_feat2', 'Q12_feat3', 'Q12_feat5', 'Q13_feat1', 'Q13_feat2', 'Q13_feat4' ]
'''
			
head = [ 'Q1_feat1', 'Q1_feat2', 'Q1_feat3', 'Q1_feat4', 'Q1_feat5', 'Q1_feat7', 'Q2_feat1', 'Q2_feat2', 'Q2_feat3', 'Q2_feat5', 'Q3_feat1', 'Q3_feat2', 
		  'Q3_feat3', 'Q3_feat4', 'Q3_feat6', 'Q4_feat1', 'Q5_feat1', 'Q5_feat2', 'Q5_feat3', 'Q5_feat4', 'Q5_feat6', 'Q6_feat1', 'Q6_feat2',
		  'Q6_feat3', 'Q6_feat4', 'Q9_feat1', 'Q9_feat2','Q9_feat3', 'Q9_feat4', 'Q9_feat6', 'Q10_feat1', 'Q10_feat2', 'Q10_feat3', 'Q10_feat4', 'Q10_feat6',
			'Q10_feat7', 'Q12_feat1', 'Q12_feat2', 'Q12_feat3', 'Q12_feat5', 'Q13_feat1', 'Q13_feat2', 'Q13_feat4' ]
			
list_smart_speakers_remove = ['Smart_Speaker_Q10_feat6', 'Smart_Speaker_Q10_feat7','Smart_Speaker_Q13_feat1']
			
#definitions needed to do participant analysis
big_table = []
temp_list = []
counter = 0 	
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
	scenario_heads = ['location_1','location_2','Relationship_1','Relationship_2']
	
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
		
	if device == 'Smart_Speaker':
		training_head = [x for x in training_head if x not in list_smart_speakers_remove]
		
		
	# if data_model_ind == 11:
		# file = 'selected_features_rfecv/final_feature_set_'+filename+"_"+Data_models[data_model_ind]+"_"+'selectkbest'+"_"+device+".csv"
		# feat_data = pd.read_csv(file)
		
		# training_head = feat_data['selected_features'] # load the features selected from rfecv
		# #print(training_head)
		
	print('the current model:',cur_model)
	print('the number of features in cur model', len(training_head))
	
	
	############ SET the resampling ###########################################################
	
	
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
											n_samples=data_points)     # to match minority class # reproducible results

	data = pd.concat([data_downsampled, alt_data])
			
	

	x_data = data[training_head]

	x_data = x_data.values.astype(float)
	
	

	y_data = data["actual_use"]
	
	
	y_data = y_data.values.astype(float)
	
	################################################################################################################
	
	#apply RFECV:
	estimator = SVC(kernel="linear",probability=True) 
	#clf1 = SVC(kernel="linear",probability=True)
	if classifier == "Decisiontree":
		estimator = tree.DecisionTreeClassifier(max_depth=3)
		#clf1 = tree.DecisionTreeClassifier()
	elif classifier == "SVM" :
		estimator = SVC(kernel="linear",probability=True)
		#clf1 = SVC(kernel="linear",probability=True)
	elif classifier == "RandomForest":
		estimator = RandomForestClassifier(max_depth=2)
		#clf1 = RandomForestClassifier()
	elif classifier == "Adaboost":
		estimator = AdaBoostClassifier()
		#clf1 = AdaBoostClassifier()
	else:
		estimator = LogisticRegression()
		#clf1 = LogisticRegression()
		
		
	rfecv = RFECV(estimator, step=1, cv=StratifiedKFold(2))
	
	rfecv.fit(x_data, y_data)
	print('number of features selected:',rfecv.n_features_)
	
	x_new = rfecv.transform(x_data)
	
	#### Extract the important features ###########
	#selected_inds = rfecv.get_support(indices=True)
	#feat_coefs = rfecv.estimator_.coef_
	
	#print(feat_coefs)

	#selected_feats = [training_head[ind] for ind in selected_inds]
	#selected_vals = list(zip(selected_feats, feat_coefs[0]))
	
	#feature_frame_selected = pd.DataFrame(selected_vals, columns=['selected_features','Coefficients'])
	#feature_frame_selected = feature_frame_selected.sort_values(["ranking"],ascending=False)
	#feat_file = 'selected_features_rfecv_finalresults_1_19_19/withmetrics/final_feature_set_'+filename+"_"+cur_model+"_"+device+".csv"
	#feature_frame_selected.to_csv(feat_file)
	
	
	
	
	### Bring in classifiers #####################
	#print('shape of data:', x_new.shape)
	
	#check for class imbalance in train & test_size
	print('positive class in data', np.sum(y_data)/y_data.shape[0])
	
	#get results directly from rfecv:
	y_pred = rfecv.estimator_.predict(x_new)
	pred_score_f1 = f1_score(y_data, y_pred)
	y_pred_prob = rfecv.estimator_.predict_proba(x_new)
	y_pred_prob = [p[1] for p in y_pred_prob]
	pred_score_auc = roc_auc_score(y_data,y_pred_prob)
	pred_score_precision = precision_score(y_data,y_pred)
	pred_score_recall = recall_score(y_data,y_pred)
	score = accuracy_score(y_data,y_pred)
	
	

	print("")
	print('current classifier:' ,classifier)
	print('F1 score obtained:', pred_score_f1)
	print('auc socre obtained', pred_score_auc)
	print('precision score obtained', pred_score_precision)
	print('recall score obtained', pred_score_recall)
	print('accuracy score', score)
	
	pred_col_name = "Predictions_"+device
	original_col_name = "actual_use_"+device
	data[pred_col_name] = y_pred
	
	# #print(list(data))
	final_data = data[['ResponseId','actual_use',pred_col_name]]
	final_data.columns = ['ResponseId',original_col_name,pred_col_name]
	
	if counter == 0: #first device
		big_table = final_data
	else:
		big_table = pd.merge(final_data,big_table,on='ResponseId')
		
	temp_list.append(device)
		
	counter+=1 #update device counter
	
	#final_data.to_csv("prediction_results/predictions_"+filename+"_"+cur_model+"_"+device+"_"+classifier+".csv")
	
	
print('#####################################################')

print('participant accuracy:', participant_analysis(big_table,temp_list))
	
	
	
	
	

	
			

			