#numpy and pandas initialization
import numpy as np
import pandas as pd
PlayTennis=pd.read_csv('Play.csv')

PlayTennis
from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()

PlayTennis['outlook'] = Le.fit_transform(PlayTennis['outlook'])
PlayTennis['humidity'] = Le.fit_transform(PlayTennis['humidity'])
PlayTennis['windy'] = Le.fit_transform(PlayTennis['Windy'])
PlayTennis['play'] = Le.fit_transform(PlayTennis['Play'])

PlayTennis

y = PlayTennis['play']
X = PlayTennis.drop(['play'],axis=1)
# Fitting the model
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, y)
# We can visualize the tree using tree.plot_tree
tree.plot_tree(clf)

import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph

# The predictions are stored in X_pred
X_pred = clf.predict(X)
# verifying if the model has predicted it all right.
X_pred == y
