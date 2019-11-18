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
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
import sys


#resample
	
from sklearn.utils import resample

from patsy import dmatrices

sys.setrecursionlimit(3000)

#import data 
datapath = 'Datasets/Encoded_files/'
filename = 'Affordance_November4_alldata'
device_list = ['Laptop', 'Smart_Phone', 'Desktop_Computer','Tablet','Smart_Speaker','Smart_Watch']
classifier_list = ['Decisiontree', 'SVM','RandomForest','Adaboost','logistic']
classifier_index = 0
classifier = classifier_list[classifier_index]
reduction_list = ['tree','variancethreshold','selectkbest','pca']
reduction_list_ind = 0
reduction = reduction_list[reduction_list_ind]

head = [ 'Q1_feat1', 'Q1_feat2', 'Q1_feat3', 'Q1_feat4', 'Q1_feat5', 'Q1_feat7', 'Q2_feat1', 'Q2_feat2', 'Q2_feat3', 'Q2_feat5', 'Q3_feat1', 'Q3_feat2', 
			'Q3_feat3', 'Q3_feat4', 'Q3_feat6', 'Q4_feat1', 'Q4_feat3', 'Q5_feat1', 'Q5_feat2', 'Q5_feat3', 'Q5_feat4', 'Q5_feat6', 'Q6_feat1', 'Q6_feat2',
			'Q6_feat3', 'Q6_feat4', 'Q7_feat1', 'Q7_feat2', 'Q7_feat3', 'Q7_feat5', 'Q8_feat1', 'Q8_feat2', 'Q8_feat3', 'Q8_feat5', 'Q9_feat1', 'Q9_feat2',
			'Q9_feat3', 'Q9_feat4', 'Q9_feat6', 'Q10_feat1', 'Q10_feat2', 'Q10_feat3', 'Q10_feat4', 'Q10_feat6', 'Q10_feat7', 'Q11_feat1', 'Q11_feat2', 
			'Q11_feat3', 'Q11_feat5', 'Q12_feat1', 'Q12_feat2', 'Q12_feat3', 'Q12_feat5', 'Q13_feat1', 'Q13_feat2', 'Q13_feat4' ]


for device in device_list:
    print("##########################################")
    print("current device dataset:", device)
    file = '~/Dropbox/SYR_GAship/afforadance_Study/Datasets/Encoded_files/encoded_Affordance_November4_alldata_'+device+'_data.csv'
    data = pd.read_csv(file)

 
    training_head = []
    training_head.extend(head)
    for e in head:
        cur = device+"_"+e
        training_head.append(cur)
        


    #adding in scenario_variables:
    scenario_heads = ['location_1','location_2','location_3','Relationship_1','Relationship_2','Relationship_3']
	
    mystr = "actual_use ~ "
    for scenario in scenario_heads:
        for feat in training_head:
            mystr = mystr + scenario + ':' + feat +'+'
            #mystr = mystr + scenario + '*' + feat +'+'
		
	
	
    training_head.extend(scenario_heads)


    x_data = data[training_head]

    #x_data = x_data.values.astype(float)



    y_data = data["actual_use"]
    
            
    mystr = mystr[:len(mystr)-1]
    
    mydata = pd.concat([x_data,y_data],axis=1)
    
    #print(mystr)
    
    y,X_values = dmatrices(mystr,mydata,return_type="dataframe")
    
    X_values = X_values.drop('Intercept',axis=1)
	
    
	
    #file_check = "interaction features_"+device+".csv"
    #X.to_csv(file_check)
	
    print('shape of X features',X_values.shape)
    print('shape of Y variables', y.shape)
	

    headers = list(X_values)
	
    #X= X.values.astype(float)
    y_data = data["actual_use"]
	
    #y = y.values.astype(float)
	
    print(X_values.head())
	
    X_train, X_test, y_train, y_test = train_test_split(X_values, y_data, test_size=0.4, random_state=123)
	
    if reduction == "tree":
        clf_feat = ExtraTreesClassifier(n_estimators=1)
        clf_feat.fit(X_train, y_train)
		#model = SelectFromModel(clf_feat)
        model = RFE(clf_feat,10, step=5) #RFE and SelectfromModel differ only in the threshold for removal.
        X_train = model.fit_transform(X_train,y_train)
        X_test = model.transform(X_test)
        feature_imp = clf_feat.feature_importances_
        feature_map = {}
       
        for ind in range(0,len(headers)):
            feature_map[headers[ind]] = feature_imp[ind]
        feature_map = sorted(feature_map.items(), key=lambda kv: kv[1],reverse=True)
        feature_frame = pd.DataFrame(feature_map,columns=['features','Importance'])
        feature_frame = feature_frame.sort_values(["Importance"],ascending=False)
        feat_file = 'feat_imp/'+'feature_imp_'+filename+"_"+reduction+"_"+"interactions_fullmodel"+device+".csv"
        feature_frame.to_csv(feat_file)

		
    elif reduction == "variancethreshold":
        sel = VarianceThreshold(threshold=(0.4 * (1 - 0.4)))
        X_train = sel.fit_transform(X_train)
        X_test = sel.transform(X_test)
		
    elif reduction == "selectkbest":
        test = SelectKBest(score_func=chi2,k=10)
        #test = SelectKBest()
        #fit = test.fit(X_train,y_train)
        X_train = test.fit_transform(X_train,y_train)
        X_test = test.transform(X_test)
        feature_imp = test.pvalues_
		

        feature_map = {}
       
        for ind in range(0,len(headers)):
            feature_map[headers[ind]] = feature_imp[ind]
        feature_map = sorted(feature_map.items(), key=lambda kv: kv[1],reverse=True)
        feature_frame = pd.DataFrame(feature_map,columns=['features','Importance']).fillna(-1)
        feature_frame = feature_frame.sort_values(["Importance"],ascending=False)
        feat_file = 'feat_imp/'+'feature_imp_'+filename+"_"+reduction+"_"+"pvalues"+device+".csv"
        feature_frame.to_csv(feat_file)

    elif reduction == "pca":
        pca = PCA(n_components=5, svd_solver='full')
        X_train = pca.fit_transform(X_train,y_train)
        X_test = pca.transform(X_test)
        pca_variance = pca.explained_variance_ratio_

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
        clf1 = LogisticRegression()
	
	
    
    cross_val_list = cross_val_score(LogisticRegression(), X_values, y_data, cv=5)
	
    cross_val_score = cross_val_list.mean()
	# clf1.fit(X_train, y_train)
	
    # y_pred = clf1.predict(X_test)
    # pred_score_f1 = f1_score(y_test, y_pred)
    # y_pred_prob = clf1.predict_proba(X_test)
    # y_pred_prob = [p[1] for p in y_pred_prob]
    # pred_score_auc = roc_auc_score(y_test,y_pred_prob)
    # pred_score_precision = precision_score(y_test,y_pred)
    # pred_score_recall = recall_score(y_test,y_pred)
    # score = clf1.score(X_test, y_test)

    print("")
    print('current classifier:' ,classifier)
    print('current reduction:', reduction)
    #print('F1 score obtained:', pred_score_f1)
    #print('auc socre obtained', pred_score_auc)
    #print('precision score obtained', pred_score_precision)
    #print('recall score obtained', pred_score_recall)
    #print('accuracy score', score)
    print('cross_val_Score', cross_val_score)
    #print('labels :', np.unique(y_pred))
    #if reduction == "pca":
     #  print('In pca explained variance is:', pca_variance)
	
	
  
	
	