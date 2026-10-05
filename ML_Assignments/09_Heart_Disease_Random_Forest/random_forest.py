
import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
df=pd.read_csv("heart_disease.csv"); X=df.drop(columns="target"); y=df.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
models={"DecisionTree":DecisionTreeClassifier(max_depth=5,random_state=42),"RandomForest":RandomForestClassifier(n_estimators=250,max_depth=7,random_state=42)}
rows=[]
for name,m in models.items():
 m.fit(Xtr,ytr); pred=m.predict(Xte)
 rows.append([name,accuracy_score(yte,pred),precision_score(yte,pred),recall_score(yte,pred),f1_score(yte,pred)])
pd.DataFrame(rows,columns=["model","accuracy","precision","recall","f1"]).to_csv("outputs/model_comparison.csv",index=False)
rf=models["RandomForest"]; imp=pd.Series(rf.feature_importances_,index=X.columns).sort_values(ascending=False)
imp.to_csv("outputs/feature_importance.csv")
plt.figure(figsize=(8,5)); imp.head(10).sort_values().plot(kind="barh"); plt.title("Random Forest Feature Importance"); plt.tight_layout(); plt.savefig("outputs/feature_importance.png",dpi=160); plt.close()
print(pd.DataFrame(rows,columns=["model","accuracy","precision","recall","f1"]))
