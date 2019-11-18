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
from regressors import stats
from sklearn.linear_model import LogisticRegressionCV

#resample
	
from sklearn.utils import resample

#set seed
np.random.seed(12345)
#import data 
datapath = 'Datasets/Encoded_files/'
filename = 'encoded_Affordance_November19_alldata'
#device_list = ['Laptop', 'Smart Phone', 'Desktop Computer','Tablet','Smart Speaker','Smart Watch']

device_list = ['Laptop','Smart_Phone','Desktop_Computer','Smart_Watch','Tablet','Smart_Speaker']
#device_list = ['Smart_Speaker']
classifier_list = ['Decisiontree', 'SVM','RandomForest','Adaboost','logistic']

classifier_index = 0
classifier = classifier_list[classifier_index]

reduction_list = ['tree','variancethreshold','selectkbest','pca','none']
reduction_list_ind = 2
reduction = reduction_list[reduction_list_ind]

Data_models = ['part1', 'part1_loc','part1_rel','part1_loc_rel','part2','part2_loc','part2_rel','part2_loc_rel','part1_part2',
				'part1_part2_loc','part1_part2_rel','part1_part2_loc_rel']
data_model_ind = 11
retrieve_features = True


# head = [ 'Q1_feat1', 'Q1_feat2', 'Q1_feat3', 'Q1_feat4', 'Q1_feat5', 'Q1_feat7', 'Q2_feat1', 'Q2_feat2', 'Q2_feat3', 'Q2_feat5', 'Q3_feat1', 'Q3_feat2', 
			# 'Q3_feat3', 'Q3_feat4', 'Q3_feat6', 'Q4_feat1', 'Q4_feat3', 'Q5_feat1', 'Q5_feat2', 'Q5_feat3', 'Q5_feat4', 'Q5_feat6', 'Q6_feat1', 'Q6_feat2',
			# 'Q6_feat3', 'Q6_feat4', 'Q7_feat1', 'Q7_feat2', 'Q7_feat3', 'Q7_feat5', 'Q8_feat1', 'Q8_feat2', 'Q8_feat3', 'Q8_feat5', 'Q9_feat1', 'Q9_feat2',
			# 'Q9_feat3', 'Q9_feat4', 'Q9_feat6', 'Q10_feat1', 'Q10_feat2', 'Q10_feat3', 'Q10_feat4', 'Q10_feat6', 'Q10_feat7', 'Q11_feat1', 'Q11_feat2', 
			# 'Q11_feat3', 'Q11_feat5', 'Q12_feat1', 'Q12_feat2', 'Q12_feat3', 'Q12_feat5', 'Q13_feat1', 'Q13_feat2', 'Q13_feat4' ]
#HEADERS AFTER REMOVING Questions 7,11,8			
head = [ 'Q1_feat1', 'Q1_feat2', 'Q1_feat3', 'Q1_feat4', 'Q1_feat5', 'Q1_feat7', 'Q2_feat1', 'Q2_feat2', 'Q2_feat3', 'Q2_feat5', 'Q3_feat1', 'Q3_feat2', 
		  'Q3_feat3', 'Q3_feat4', 'Q3_feat6', 'Q4_feat1', 'Q5_feat1', 'Q5_feat2', 'Q5_feat3', 'Q5_feat4', 'Q5_feat6', 'Q6_feat1', 'Q6_feat2',
		  'Q6_feat3', 'Q6_feat4', 'Q9_feat1', 'Q9_feat2','Q9_feat3', 'Q9_feat4', 'Q9_feat6', 'Q10_feat1', 'Q10_feat2', 'Q10_feat3', 'Q10_feat4', 'Q10_feat6',
			'Q10_feat7', 'Q12_feat1', 'Q12_feat2', 'Q12_feat3', 'Q12_feat5', 'Q13_feat1', 'Q13_feat2', 'Q13_feat4' ]
			

list_smart_speakers_remove = ['Smart_Speaker_Q10_feat6', 'Smart_Speaker_Q10_feat7','Smart_Speaker_Q13_feat1']

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
											n_samples=data_points)     # to match minority class # reproducible results

	data = pd.concat([data_downsampled, alt_data])
			
	

	x_data = data[training_head]

	x_data = x_data.values.astype(float)
	
	

	y_data = data["actual_use"]
	
	
	y_data = y_data.values.astype(float)
	
	
	#tree.DecisionTreeClassifier()#LogisticRegression()
	estimator =LogisticRegression(fit_intercept=True)  #tree.DecisionTreeClassifier() #LogisticRegression()   #SVC(kernel="linear")  #tree.DecisionTreeClassifier() #LogisticRegression()#
	rfecv = RFECV(estimator, step=1, cv=StratifiedKFold(2))
	
	rfecv.fit(x_data, y_data)
	print('number of features selected:',rfecv.n_features_)
	
	x_new = rfecv.transform(x_data)
	
	selected_inds = rfecv.get_support(indices=True)
	selected_ranks = rfecv.ranking_

	selected_feats = [training_head[ind] for ind in selected_inds]
	#print(selected_feats)
	
	#print(rfecv.estimator_.coef_)
	pvals = stats.coef_pval(rfecv.estimator_, x_new, y_data)
	#print(pvals)
	
	
	cl1 = LogisticRegressionCV(cv=10,penalty='l2',fit_intercept=True)
	
	cl1.fit(x_data,y_data)
	
	coefs = cl1.coef_
	intercept = cl1.intercept_[0]
	
	pvals_cur = stats.coef_pval(cl1, x_data, y_data)

	
	all_headers = []
	all_headers.append('intercept')
	all_headers.extend(training_head)
	
	all_coefficients = []
	all_coefficients.append(intercept)
	all_coefficients.extend(coefs[0])
	
	sig_inds = np.where(pvals_cur <= 0.05)[0].tolist()
	
	pval_list = pvals_cur.tolist()
	sig_list = [(all_headers[i],all_coefficients[i],pval_list[i]) for i in sig_inds]
	
	sig_feat_frame = pd.DataFrame(sig_list,columns=['Significant features','Coefficients','p-val'])
	sig_file_path = 'significant features/'+filename+"_"+cur_model+"_"+reduction+"_"+device+".csv"
	sig_feat_frame.to_csv(sig_file_path)
	
	
	feature_frame_selected = pd.DataFrame(selected_feats,columns=['selected_features'])
	#feature_frame_selected = feature_frame_selected.sort_values(["ranking"],ascending=False)
	feat_file = 'selected_features_rfecv_decisiontree/final_feature_set_'+filename+"_"+cur_model+"_"+reduction+"_"+device+".csv"
	#feature_frame_selected.to_csv(feat_file)
	
	# clf1 = LogisticRegression()

	
	# print('shape of new dataset:',x_new.shape)
	# #y_pred = rfecv.estimator_.predict(x_new)
	# y_pred = cross_val_predict(clf1,x_new,y_data,cv=StratifiedKFold(10))
	# pred_score_f1 = f1_score(y_data, y_pred)
	# y_pred_prob = rfecv.estimator_.predict_proba(x_new)
	# y_pred_prob = [p[1] for p in y_pred_prob]
	# pred_score_auc = roc_auc_score(y_data,y_pred_prob)
	# pred_score_precision = precision_score(y_data,y_pred)
	# pred_score_recall = recall_score(y_data,y_pred)
	# score = accuracy_score(y_data,y_pred)
	
	# print("")
	# print('current classifier:' ,classifier)
	# #print('current reduction:', reduction)
	# print('F1 score obtained:', pred_score_f1)
	# print('auc socre obtained', pred_score_auc)
	# print('precision score obtained', pred_score_precision)
	# print('recall score obtained', pred_score_recall)
	# print('accuracy score', score)
	
	plt.figure()
	plt.xlabel("Number of features selected")
	title = "results for "+device
	plt.suptitle(title)
	plt.ylabel("Accuracy")
	plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
	file_name = "variations_num_features_rfecv_decisiontree/cur_plots_accuracies_with_rfecv"+filename+"_"+cur_model+"_"+reduction+"_"+device+".pdf"
	plt.savefig(file_name, bbox_inches='tight')
	plt.show()