
import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
df=pd.read_csv("heart_disease.csv"); X=df.drop(columns="target"); y=df.target
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
models={
"XGBoost":XGBClassifier(n_estimators=180,max_depth=4,learning_rate=.05,subsample=.9,colsample_bytree=.9,
                         eval_metric="logloss",random_state=42),
"LightGBM":LGBMClassifier(n_estimators=180,max_depth=5,learning_rate=.05,verbosity=-1,random_state=42)
}
rows=[]
for name,m in models.items():
 m.fit(Xtr,ytr); pred=m.predict(Xte)
 rows.append([name,accuracy_score(yte,pred),precision_score(yte,pred),recall_score(yte,pred),f1_score(yte,pred)])
pd.DataFrame(rows,columns=["model","accuracy","precision","recall","f1"]).to_csv("outputs/model_comparison.csv",index=False)
# SHAP for the better tree ensemble
best=models["XGBoost"]
explainer=shap.TreeExplainer(best)
sv=explainer.shap_values(Xte.iloc[:100])
plt.figure(); shap.summary_plot(sv,Xte.iloc[:100],show=False,max_display=12); plt.tight_layout(); plt.savefig("outputs/shap_summary_xgboost.png",dpi=160,bbox_inches="tight"); plt.close()
imp=pd.Series(abs(sv).mean(axis=0),index=X.columns).sort_values(ascending=False)
imp.to_csv("outputs/shap_mean_abs_importance.csv")
plt.figure(figsize=(8,5)); imp.head(10).sort_values().plot(kind="barh"); plt.title("Mean |SHAP| Feature Importance - XGBoost"); plt.tight_layout(); plt.savefig("outputs/shap_feature_importance.png",dpi=160); plt.close()
print(pd.DataFrame(rows,columns=["model","accuracy","precision","recall","f1"]))
