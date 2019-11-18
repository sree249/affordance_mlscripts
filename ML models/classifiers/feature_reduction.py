



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
from sklearn import cross_validation

#packages for feature selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import matplotlib.pyplot as plt



#import data
datapath = 'Datasets/Encoded_files/'
filename = 'Affordance_October27_alldata'
device_list = ['Laptop', 'Smart Phone', 'Desktop Computer','Tablet','Smart Speaker','Smart Watch']
classifier = "RandomForest"

head = [ 'Q1_feat1', 'Q1_feat2', 'Q1_feat3', 'Q1_feat4', 'Q1_feat5', 'Q1_feat7', 'Q2_feat1', 'Q2_feat2', 'Q2_feat3', 'Q2_feat5', 'Q3_feat1', 'Q3_feat2',
			'Q3_feat3', 'Q3_feat4', 'Q3_feat6', 'Q4_feat1', 'Q4_feat3', 'Q5_feat1', 'Q5_feat2', 'Q5_feat3', 'Q5_feat4', 'Q5_feat6', 'Q6_feat1', 'Q6_feat2',
			'Q6_feat3', 'Q6_feat4', 'Q7_feat1', 'Q7_feat2', 'Q7_feat3', 'Q7_feat5', 'Q8_feat1', 'Q8_feat2', 'Q8_feat3', 'Q8_feat5', 'Q9_feat1', 'Q9_feat2',
			'Q9_feat3', 'Q9_feat4', 'Q9_feat6', 'Q10_feat1', 'Q10_feat2', 'Q10_feat3', 'Q10_feat4', 'Q10_feat6', 'Q10_feat7', 'Q11_feat1', 'Q11_feat2',
			'Q11_feat3', 'Q11_feat5', 'Q12_feat1', 'Q12_feat2', 'Q12_feat3', 'Q12_feat5', 'Q13_feat1', 'Q13_feat2', 'Q13_feat4' ]

for device in device_list:
	print("##########################################")
	print("current device dataset:", device)
	file = '~/Dropbox/SYR_GAship/afforadance_Study/Datasets/Encoded_files/encoded_Affordance_October27_alldata_'+device+'_data.csv'
	data = pd.read_csv(file)



	training_head = []
	training_head.extend(head)
	for e in head:
		cur = device+"_"+e
		training_head.append(cur)


	x_data = data[training_head]

    #correlation matrix
	corr = x_data.corr()

	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# cax = ax.matshow(corr, interpolation='nearest')
	# fig.colorbar(cax)

	#plt.matshow(corr)
	#plt.xticks(range(len(corr.columns)),corr.columns)
	#plt.yticks(range(len(corr.columns)),corr.columns)
	plt.show()



	x_data = x_data.values.astype(float)



	y_data = data["actual_use"]
	y_data = y_data.values.astype(float)

	X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=123)

    #select from model
	clf_feat = ExtraTreesClassifier()
	clf_feat.fit(X_train, y_train)
	model = SelectFromModel(clf_feat, prefit=True)
	#X_train = model.transform(X_train)
	#X_test = model.transform(X_test)
	#select with variance
	sel = VarianceThreshold(threshold=(0.6 * (1 - 0.6)))

	X_train = sel.fit_transform(X_train)
	X_test = sel.transform(X_test)

	#test = SelectKBest(score_func=chi2,k=10)
	#fit = test.fit(X_train,y_train)
	#X_train = fit.transform(X_train)
	#X_test = fit.transform(X_test)


	print('shape of train data:', X_train.shape)
	print('shape of test data:', X_test.shape)

	#check for class imbalance in train & test_size
	print('positive class in train', np.sum(y_train)/y_train.shape[0])
	print('positive class in test', np.sum(y_test)/y_test.shape[0])

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
		clf1 = LogisticRegressionClassifier()


	clf1.fit(X_train, y_train)


	y_pred = clf1.predict(X_test)
	pred_score_f1 = f1_score(y_test, y_pred)
	y_pred_prob = clf1.predict_proba(X_test)
	y_pred_prob = [p[1] for p in y_pred_prob]
	pred_score_auc = roc_auc_score(y_test,y_pred_prob)
	pred_score_precision = precision_score(y_test,y_pred)
	pred_score_recall = recall_score(y_test,y_pred)
	score = clf1.score(X_test, y_test)

	print("")
	print('current classifier:' ,classifier)
	print('F1 score obtained:', pred_score_f1)
	print('auc socre obtained', pred_score_auc)
	print('precision score obtained', pred_score_precision)
	print('recall score obtained', pred_score_recall)
	print('accuracy score', score)
	print('labels :', np.unique(y_pred))
