#feature correlations/extraction approach

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
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import matplotlib.pyplot as plt



#import data
datapath = 'Datasets/Encoded_files/'
filename = 'Affordance_October27_alldata'
device_list = ['Laptop', 'Smart Phone', 'Desktop Computer','Tablet','Smart Speaker','Smart Watch']


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

	#plotting function for visualization
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(corr, interpolation='nearest')
	fig.colorbar(cax)

	#plt.matshow(corr)
	#plt.xticks(range(len(corr.columns)),corr.columns)
	#plt.yticks(range(len(corr.columns)),corr.columns)
	plt.show()
	
	#file_name = "correlations/"+filename+"_"+device+".csv"
	#corr.to_csv(file_name)
	#print(corr)