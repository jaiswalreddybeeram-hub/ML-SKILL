
import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
df=pd.read_csv("wine_quality.csv"); X=df.drop(columns="quality"); y=df.quality
# Group quality into 3 practical classes: low (<=4), medium (5-6), high (>=7)
y3=pd.cut(y,[-1,4,6,10],labels=["Low","Medium","High"])
Xtr,Xte,ytr,yte=train_test_split(X,y3,test_size=.2,stratify=y3,random_state=42)
models={"DecisionTree":DecisionTreeClassifier(max_depth=6,random_state=42),
        "RandomForest":RandomForestClassifier(n_estimators=250,max_depth=8,random_state=42)}
rows=[]
for name,m in models.items():
 m.fit(Xtr,ytr); pred=m.predict(Xte)
 rows.append([name,accuracy_score(yte,pred),precision_score(yte,pred,average="weighted"),recall_score(yte,pred,average="weighted"),f1_score(yte,pred,average="weighted")])
pd.DataFrame(rows,columns=["model","accuracy","precision_weighted","recall_weighted","f1_weighted"]).to_csv("outputs/model_comparison.csv",index=False)
rf=models["RandomForest"]; imp=pd.Series(rf.feature_importances_,index=X.columns).sort_values(ascending=False)
imp.to_csv("outputs/feature_importance.csv")
plt.figure(figsize=(8,5)); imp.head(10).sort_values().plot(kind="barh"); plt.title("Wine Quality Random Forest Feature Importance"); plt.tight_layout(); plt.savefig("outputs/feature_importance.png",dpi=160); plt.close()
cm=confusion_matrix(yte,models["RandomForest"].predict(Xte),labels=["Low","Medium","High"])
pd.DataFrame(cm,index=["Actual Low","Actual Medium","Actual High"],columns=["Pred Low","Pred Medium","Pred High"]).to_csv("outputs/confusion_matrix.csv")
print(pd.DataFrame(rows,columns=["model","accuracy","precision_weighted","recall_weighted","f1_weighted"]))
