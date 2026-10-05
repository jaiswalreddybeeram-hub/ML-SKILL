
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report,RocCurveDisplay
df=pd.read_csv("titanic.csv")
X=df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]; y=df.Survived
num=["Pclass","Age","SibSp","Parch","Fare"]; cat=["Sex","Embarked"]
pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),
                       ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)])
model=Pipeline([("pre",pre),("clf",LogisticRegression(max_iter=2000))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
model.fit(Xtr,ytr); pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
metrics=pd.DataFrame([{"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred),"recall":recall_score(yte,pred),"f1":f1_score(yte,pred),"roc_auc":roc_auc_score(yte,prob)}])
metrics.to_csv("outputs/metrics.csv",index=False)
pd.DataFrame(confusion_matrix(yte,pred),index=["Actual 0","Actual 1"],columns=["Pred 0","Pred 1"]).to_csv("outputs/confusion_matrix.csv")
RocCurveDisplay.from_predictions(yte,prob); plt.title("Titanic Logistic Regression ROC"); plt.tight_layout(); plt.savefig("outputs/roc_curve.png",dpi=160); plt.close()
print(metrics.to_string(index=False))
print(classification_report(yte,pred))
