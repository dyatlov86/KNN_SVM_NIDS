import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,confusion_matrix,roc_auc_score,make_scorer,recall_score,precision_score,ConfusionMatrixDisplay
packets = pd.read_csv('updated_dataset_multiclass.csv')
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
select=SelectKBest(mutual_info_classif,k=8)
select.fit(X_train,y_train)
print(X_train.columns[select.get_support()])

feature_names_packets=X_train.columns[select.get_support()]

 #potential classes

packets = pd.read_csv('updated_dataset_multiclass.csv')
#X = data.iloc[:, :-1].values
#y = data.iloc[:, -1].values
packets.head()

X_packets = packets[feature_names_packets] #setting the col names
y_packets = packets['label'] #setting the col names




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict
import seaborn as sns
from sklearn.svm import LinearSVC
k = 200
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

for i in range(10,k+2,1):
    print(str(i)+"\n","eğitim başladı")
    svm_model = LinearSVC(C=i,dual=False,max_iter=12000)
    print("puanlama başladı")
    y_pred = cross_val_predict(svm_model, X_packets, y_packets, cv=10,n_jobs=5)
    f1scores = cross_val_score(svm_model, X_packets, y_packets, cv=10,scoring="f1_macro",n_jobs=5)

    acscores = cross_val_score(svm_model, X_packets, y_packets, cv=10,scoring="accuracy",n_jobs=5)
    recscores = cross_val_score(svm_model, X_packets, y_packets, cv=10,scoring="recall_macro",n_jobs=5)
    prescores = cross_val_score(svm_model, X_packets, y_packets, cv=10,scoring="precision_macro",n_jobs=5)
    cm = confusion_matrix(y_packets, y_pred,labels=["anormal","normal"])
    cm_df = pd.DataFrame(cm,index =["anormal","normal"] , columns = ["anormal","normal"])
    print(cm_df)

    #cm = confusion_matrix(y_train, y_pred)
    #cm_df = pd.DataFrame(cm,index =["asreproasting","ddos","exploit-samba","kerberos-enum","nmap","normal","samba-enum"] , columns = ["asreproasting","ddos","exploit-samba","kerberos-enum","nmap","normal","samba-enum"])
    #plt.figure(figsize=(9,8))
    #sns.heatmap(cm_df, annot=True)
    #plt.title('Confusion Matrix')
    #plt.ylabel('Actal Values')
    #plt.xlabel('Predicted Values')
    #plt.show()
    

    fs.append(np.mean(f1scores))
    acs.append(np.mean(acscores))
    print(np.mean(acscores))
    recs.append(np.mean(recscores))
    pres.append(np.mean(prescores))
#    fs.append(cross_val_score(svm_model, X_packets, y_packets, scoring=fscorer,error_score="raise").mean())
#    pres.append(cross_val_score(svm_model, X_packets, y_packets, scoring=prescorer,error_score="raise").mean())
#    recs.append(cross_val_score(svm_model, X_packets, y_packets, scoring=recscorer,error_score="raise").mean())
#    acs.append(cross_val_score(svm_model, X_packets, y_packets,  scoring="accuracy",error_score="raise").mean())
    ks.append(i)

print("max:",max(acs))
print(ks,acs)
plt.plot(ks,acs,label="accuracy")
plt.plot(ks,recs,label="recall")
plt.plot(ks,fs,label="f1")
plt.plot(ks,pres,label="precision")
#plt.plot(ks,graph_y,label="score")
plt.ylabel("Score")
plt.xlabel("C value")
plt.legend()
plt.show()
#x_new = scaler.transform([[3, 8]])
#y_pred = svm_model.predict(x_new)
#print(f"Tahmin Edilen Sınıf: {y_pred[0]}")

