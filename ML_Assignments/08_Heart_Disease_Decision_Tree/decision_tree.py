
import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
df=pd.read_csv("heart_disease.csv"); X=df.drop(columns="target"); y=df.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
model=DecisionTreeClassifier(max_depth=5,random_state=42); model.fit(Xtr,ytr); pred=model.predict(Xte)
pd.DataFrame([{"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred),"recall":recall_score(yte,pred),"f1":f1_score(yte,pred)}]).to_csv("outputs/metrics.csv",index=False)
cm=confusion_matrix(yte,pred); pd.DataFrame(cm).to_csv("outputs/confusion_matrix.csv",index=False)
plt.figure(figsize=(18,9)); plot_tree(model,feature_names=X.columns,class_names=["No disease","Disease"],filled=True,max_depth=4,fontsize=7); plt.tight_layout(); plt.savefig("outputs/decision_tree.png",dpi=150); plt.close()
print(classification_report(yte,pred))
