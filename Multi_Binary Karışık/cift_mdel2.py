import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,confusion_matrix,roc_auc_score,make_scorer,recall_score,precision_score,ConfusionMatrixDisplay
packets = pd.read_csv('attack.csv')
#X = data.iloc[:, :-1].values
#y = data.iloc[:, -1].values
packets.head()
feature_names_packets = ['ip.flags.df','ip.flags.mf','ip.fragment','ip.fragment.count','ip.fragments','ip.ttl','ip.proto','tcp.window_size','tcp.ack','tcp.seq','tcp.len','tcp.stream','tcp.urgent_pointer','tcp.flags','tcp.analysis.ack_rtt','tcp.segments','tcp.reassembled.length','http.request','udp.port','frame.time_relative','frame.time_delta','tcp.time_relative','tcp.time_delta','tcp.port']
X_packets = packets[feature_names_packets] #setting the col names
y_packets = packets['label'] #setting the col names
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_packets,y_packets,test_size=0.25)
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import mutual_info_classif,chi2,f_classif,SelectFpr,SelectPercentile
select=SelectKBest(mutual_info_classif,k=7)
select.fit(X_train,y_train)
print(X_train.columns[select.get_support()])

feature_names_packets=X_train.columns[select.get_support()]

packets = pd.read_csv('attack.csv')
#X = data.iloc[:, :-1].values
#y = data.iloc[:, -1].values
packets.head()

X_packets = packets[feature_names_packets] #setting the col names
y_packets = packets['label'] #setting the col names



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X_packets, y_packets, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict
import seaborn as sns
k=11
fs=[]
ks=[]
pres=[]
recs=[]
acs=[]
graph_y=[]
#print(y_train.value_counts())

#k=11

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train)



dataattack = pd.read_csv('normal2.csv')
#X = data.iloc[:, :-1].values
#y = data.iloc[:, -1].values
packets.head()

X_packets = dataattack[["tcp.window_size","tcp.ack","tcp.stream","tcp.flags","frame.time_relative","tcp.time_relative","tcp.port"]] #setting the col names
y_packets = dataattack['label'] #setting the col names


file1 = open('normal2.csv', 'r')
count = 0
 
#while True:
#    count += 1
 
    # Get next line from file
#    line = file1.readline()
#    line=line.replace("\n","")
#    ex=line.split(",")
#    print(ex)
x_new = scaler.transform(X_packets)
y_pred = knn.predict(x_new)
print(y_pred)
anormal=0
normal=0
index=1
for i in y_pred:
    if i =="normal":
        anormal=anormal+1
    else:
        normal=normal+1
        print(index,y_pred[index-1])
    index=index+1
print("anormal:",anormal,"normal:",normal)
    # if line is empty
    # end of file is reached
#    if not line:
#        break
    
