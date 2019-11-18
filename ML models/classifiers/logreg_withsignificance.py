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


#HEADERS AFTER REMOVING Questions 7,11,8			
head = [ 'Q1_feat1', 'Q1_feat2', 'Q1_feat3', 'Q1_feat4', 'Q1_feat5', 'Q1_feat7', 'Q2_feat1', 'Q2_feat2', 'Q2_feat3', 'Q2_feat5', 'Q3_feat1', 'Q3_feat2', 
		  'Q3_feat3', 'Q3_feat4', 'Q3_feat6', 'Q4_feat1', 'Q5_feat1', 'Q5_feat2', 'Q5_feat3', 'Q5_feat4', 'Q5_feat6', 'Q6_feat1', 'Q6_feat2',
		  'Q6_feat3', 'Q6_feat4', 'Q9_feat1', 'Q9_feat2','Q9_feat3', 'Q9_feat4', 'Q9_feat6', 'Q10_feat1', 'Q10_feat2', 'Q10_feat3', 'Q10_feat4', 'Q10_feat6',
			'Q10_feat7', 'Q12_feat1', 'Q12_feat2', 'Q12_feat3', 'Q12_feat5', 'Q13_feat1', 'Q13_feat2', 'Q13_feat4' ]