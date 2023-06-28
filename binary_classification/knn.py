import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,confusion_matrix,roc_auc_score,make_scorer,recall_score,precision_score,ConfusionMatrixDisplay
packets = pd.read_csv('updated_dataset_multiclass.csv')
#X = data.iloc[:, :-1].values
#y = data.iloc[:, -1].values

feature_names_packets=['ip.ttl', 'ip.proto', 'tcp.window_size', 'tcp.ack', 'tcp.seq','tcp.len', 'tcp.stream', 'tcp.flags', 'tcp.analysis.ack_rtt','udp.port', 'frame.time_relative', 'frame.time_delta','tcp.time_relative', 'tcp.time_delta', 'tcp.port']
X_packets = packets[feature_names_packets] #setting the col names
y_packets = packets['label'] #setting the col names
 #potential classes

packets.head()




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X_packets, y_packets, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict
import seaborn as sns
k = 75
fs=[]
ks=[]
pres=[]
recs=[]
acs=[]
graph_y=[]
#print(y_train.value_counts())

#prescorer=make_scorer(precision_score,labels=y_packets,average="macro")
#recscorer=make_scorer(recall_score,labels=y_packets,average="micro")
#fscorer=make_scorer(f1_score,labels=y_packets,average="weighted")

for i in range(3,k+2,2):
    print(str(i)+"\n")
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    #y_pred = cross_val_predict(knn, X_train, y_train, cv=2)
#    f1scores = cross_val_score(knn, X_packets, y_packets, cv=5,scoring="f1_macro")
    acscores = cross_val_score(knn, X_packets, y_packets, cv=10,scoring="accuracy",n_jobs=4)
#    recscores = cross_val_score(knn, X_packets, y_packets, cv=5,scoring="recall_macro")
#    prescores = cross_val_score(knn, X_packets, y_packets, cv=10,scoring="precision_macro")
    #cm = confusion_matrix(y_train, y_pred)
    #cm_df = pd.DataFrame(cm,index =["asreproasting","ddos","exploit-samba","kerberos-enum","nmap","normal","samba-enum"] , columns = ["asreproasting","ddos","exploit-samba","kerberos-enum","nmap","normal","samba-enum"])
    #plt.figure(figsize=(9,8))
    #sns.heatmap(cm_df, annot=True)
    #plt.title('Confusion Matrix')
    #plt.ylabel('Actal Values')
    #plt.xlabel('Predicted Values')
    #plt.show()
    

#    fs.append(np.mean(f1scores))
    acs.append(np.mean(acscores))
#    recs.append(np.mean(recscores))
#    pres.append(np.mean(prescores))
#    fs.append(cross_val_score(knn, X_packets, y_packets, scoring=fscorer,error_score="raise").mean())
#    pres.append(cross_val_score(knn, X_packets, y_packets, scoring=prescorer,error_score="raise").mean())
#    recs.append(cross_val_score(knn, X_packets, y_packets, scoring=recscorer,error_score="raise").mean())
#    acs.append(cross_val_score(knn, X_packets, y_packets,  scoring="accuracy",error_score="raise").mean())
    ks.append(i)


print(ks,acs)
plt.plot(ks,acs,label="accuracy")
#plt.plot(ks,recs,label="recall")
#plt.plot(ks,fs,label="f1")
#plt.plot(ks,pres,label="precision")
#plt.plot(ks,graph_y,label="score")
plt.ylabel("Accuracy")
plt.xlabel("k value")
plt.legend()
plt.show()
#x_new = scaler.transform([[3, 8]])
#y_pred = knn.predict(x_new)
#print(f"Tahmin Edilen Sınıf: {y_pred[0]}")

