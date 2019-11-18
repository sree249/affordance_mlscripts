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
num_feat = 10
Data_models = ['part1', 'part1_loc','part1_rel','part1_loc_rel','part2','part2_loc','part2_rel','part2_loc_rel','part1_part2',
				'part1_part2_loc','part1_part2_rel','part1_part2_loc_rel']
data_model_ind = 11
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
	
	X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=123)
	
	if reduction == "tree":
		clf_feat = ExtraTreesClassifier(n_estimators=1)
		clf_feat.fit(x_data, y_data)
		#model = SelectFromModel(clf_feat)
		model = RFE(clf_feat, num_feat, step=1) #RFE and SelectfromModel differ only in the threshold for removal.
		x_data = model.fit_transform(x_data,y_data)
		feature_imp = clf_feat.feature_importances_
		feature_map = {}
		for ind in range(0,len(training_head)):
			feature_map[training_head[ind]] = feature_imp[ind]
		feature_map = sorted(feature_map.items(), key=lambda kv: kv[1],reverse=True)
		feature_frame = pd.DataFrame(feature_map,columns=['features','Importance'])
		feature_frame = feature_frame.sort_values(["Importance"],ascending=False)
		feat_file = 'feat_imp/'+'feature_imp_withcrossvalidation_'+filename+"_"+reduction+"_"+device+".csv"
		#feature_frame.to_csv(feat_file)
		
	elif reduction == "selectkbest":
		test = SelectKBest(score_func=chi2,k=num_feat)
		#RFE(clf_feat, num_feat, step=1) 
		fit = test.fit(x_data,y_data)
		x_data = fit.transform(x_data)
		feature_imp = test.scores_
		feature_map = {}
		for ind in range(0,len(training_head)):
			feature_map[training_head[ind]] = feature_imp[ind]
		#feature_map = sorted(feature_map.items(), key=lambda kv: kv[1],reverse=True)
		feature_frame = pd.DataFrame(list(feature_map.items()),columns=['features','Importance'])
		feature_frame = feature_frame.sort_values(["Importance"],ascending=False)
		feat_file = 'feat_imp/'+'feature_imp_withcrossvalidation_'+filename+"_"+reduction+"_"+device+".csv"
		#feature_frame.to_csv(feat_file)
	
		selected_inds = test.get_support(indices=True)
		selected_feats = [training_head[ind] for ind in selected_inds]
		#print(selected_feats)
		feat_map_select = {}
		for ele in selected_feats:
			feat_map_select[ele] = feature_map[ele]
		feature_frame_selected = pd.DataFrame(list(feat_map_select.items()),columns=['features','Importance'])
		feature_frame_selected = feature_frame_selected.sort_values(["Importance"],ascending=False)
		feat_file = 'selected_features/final_feature_set_'+filename+"_"+cur_model+"_"+reduction+"_"+device+".csv"
		feature_frame_selected.to_csv(feat_file)
		
		
	# elif reduction == "logistic":
		# clf_feat =LogisticRegression()
		# clf_feat.feat(x_data,y_data)
		# model = RFE(clf_feat, num_feat,step=1)
		# x_data = model
		
	print('shape of data:', x_data.shape)
	
	#check for class imbalance in train & test_size
	print('positive class in data', np.sum(y_data)/y_data.shape[0])
	
	#default classifer = logistic regression
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
	
	#adding 10 fold cross-validation
	# pred_score_f1 = cross_val_score(clf1, x_data, y_data, cv=10,scoring='f1').mean()
	# pred_score_auc = cross_val_score(clf1, x_data, y_data, cv=10,scoring='roc_auc').mean()
	# pred_score_precision = cross_val_score(clf1, x_data, y_data, cv=10,scoring='precision').mean()
	# pred_score_recall = cross_val_score(clf1, x_data, y_data, cv=10,scoring='recall').mean()
	# score = cross_val_score(clf1, x_data, y_data, cv=10).mean()
	
	
	# clf1.fit(X_train, y_train)
	
	
	
	y_pred = cross_val_predict(clf1, x_data, y_data, cv=10)
	pred_score_f1 = f1_score(y_data, y_pred)
	y_pred_prob = cross_val_predict(clf1, x_data, y_data, cv=10,method='predict_proba')
	y_pred_prob = [p[1] for p in y_pred_prob]
	pred_score_auc = roc_auc_score(y_data,y_pred_prob)
	pred_score_precision = precision_score(y_data,y_pred)
	pred_score_recall = recall_score(y_data,y_pred)
	score = accuracy_score(y_data,y_pred)
	

	print("")
	print('current classifier:' ,classifier)
	print('current reduction:', reduction)
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
	
	#final_data.to_csv("predictions_"+device+".csv")
	
	


	