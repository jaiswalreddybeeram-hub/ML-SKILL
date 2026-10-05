
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,classification_report
df=pd.read_csv("diabetes_severity.csv")
X=df.drop(columns="Severity"); y=df.Severity
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
model=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),
                ("clf",LogisticRegression(max_iter=2000))])
model.fit(Xtr,ytr); pred=model.predict(Xte)
pr,re,f1,_=precision_recall_fscore_support(yte,pred,average="weighted")
pd.DataFrame([{"accuracy":accuracy_score(yte,pred),"precision_weighted":pr,"recall_weighted":re,"f1_weighted":f1}]).to_csv("outputs/metrics.csv",index=False)
cm=confusion_matrix(yte,pred,labels=["Mild","Moderate","Severe"])
pd.DataFrame(cm,index=["Actual Mild","Actual Moderate","Actual Severe"],columns=["Pred Mild","Pred Moderate","Pred Severe"]).to_csv("outputs/confusion_matrix.csv")
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Mild","Moderate","Severe"],yticklabels=["Mild","Moderate","Severe"])
plt.title("Diabetes Severity Confusion Matrix"); plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png",dpi=160); plt.close()
print(classification_report(yte,pred))
