import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt

oku=pandas.read_csv("updated_dataset_multiclass.csv",sep=",")
feature_names_packets = ['ip.ttl', 'tcp.window_size', 'tcp.ack', 'tcp.seq', 'tcp.stream','frame.time_relative', 'tcp.time_relative', 'tcp.port']
y=oku["label"]
X=oku[feature_names_packets]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import mutual_info_classif,chi2,f_classif,SelectFpr,SelectPercentile
select=SelectKBest(mutual_info_classif,k=8)
select.fit(X_train,y_train)
print(X_train.columns[select.get_support()])
