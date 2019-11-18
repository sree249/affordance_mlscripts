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
#from sklearn import cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE

reduction_list = ['tree','variancethreshold','selectkbest','pca','none']
reduction_list_ind = 0
reduction = reduction_list[reduction_list_ind]
num_feat = 15

from graphviz import Source



head = [ 'Q1_feat1', 'Q1_feat2', 'Q1_feat3', 'Q1_feat4', 'Q1_feat5', 'Q1_feat7', 'Q2_feat1', 'Q2_feat2', 'Q2_feat3', 'Q2_feat5', 'Q3_feat1', 'Q3_feat2', 
			'Q3_feat3', 'Q3_feat4', 'Q3_feat6', 'Q4_feat1', 'Q4_feat3', 'Q5_feat1', 'Q5_feat2', 'Q5_feat3', 'Q5_feat4', 'Q5_feat6', 'Q6_feat1', 'Q6_feat2',
			'Q6_feat3', 'Q6_feat4', 'Q7_feat1', 'Q7_feat2', 'Q7_feat3', 'Q7_feat5', 'Q8_feat1', 'Q8_feat2', 'Q8_feat3', 'Q8_feat5', 'Q9_feat1', 'Q9_feat2',
			'Q9_feat3', 'Q9_feat4', 'Q9_feat6', 'Q10_feat1', 'Q10_feat2', 'Q10_feat3', 'Q10_feat4', 'Q10_feat6', 'Q10_feat7', 'Q11_feat1', 'Q11_feat2', 
			'Q11_feat3', 'Q11_feat5', 'Q12_feat1', 'Q12_feat2', 'Q12_feat3', 'Q12_feat5', 'Q13_feat1', 'Q13_feat2', 'Q13_feat4' ]


device_list = ['Laptop','Smart_Phone','Desktop_Computer','Smart_Watch','Tablet','Smart_Speaker']
dev_ind = 1
device = device_list[dev_ind]

file = 'C:/Users/sree2/Dropbox/SYR_GAship/afforadance_Study/Datasets/Encoded_files/encoded_Affordance_November19_alldata_'+device+'_data.csv'
data = pd.read_csv(file)

training_head = []
training_head.extend(head)
for e in head:
	cur = device+"_"+e
	training_head.append(cur)
		
	
#adding in scenario_variables:
scenario_heads = ['location_1','location_2','location_3','Relationship_1','Relationship_2','Relationship_3']
	
training_head.extend(scenario_heads)

	
x_data = data[training_head]

x_data = x_data.values.astype(float)
	
	

y_data = data["actual_use"]

	
test = SelectKBest(score_func=chi2,k=num_feat)
fit = test.fit(x_data,y_data)
x_data = fit.transform(x_data)
feature_imp = test.scores_
feature_map = {}
for ind in range(0,len(training_head)):
	feature_map[training_head[ind]] = feature_imp[ind]
	
	
selected_inds = test.get_support(indices=True)
selected_feats = [training_head[ind] for ind in selected_inds]



print('shape of data:', x_data.shape)
	
#check for class imbalance in train & test_size
print('positive class in data', np.sum(y_data)/y_data.shape[0])

clf1 = tree.DecisionTreeClassifier(criterion="gini",random_state=0,max_depth=20,min_samples_leaf=5)
	
clf1.fit(x_data, y_data)

#tree_data = clf1.tree_

#print(tree_data.min_samples_leaf)

tree.export_graphviz(clf1,out_file='C:/Users/sree2/Dropbox/SYR_GAship/afforadance_Study/ML models/test_viz.dot',feature_names=selected_feats,filled = True) 

cur_file_name = 'tree_visual_'+str(num_feat)+"_"+device+".png"
graph = Source( tree.export_graphviz(clf1, out_file=None, feature_names=selected_feats))
png_bytes = graph.pipe(format='png')
with open(cur_file_name,'wb') as f:
    f.write(png_bytes)
	
	



	

