
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
df=pd.read_csv("california_housing.csv"); out="outputs"
X=df.drop(columns="MedHouseVal"); y=df.MedHouseVal
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
models={
"LinearRegression":LinearRegression(),
"Ridge":Pipeline([("scaler",StandardScaler()),("model",Ridge(alpha=1.0))]),
"Lasso":Pipeline([("scaler",StandardScaler()),("model",Lasso(alpha=.01,max_iter=10000))])
}
rows=[]
for name,m in models.items():
 m.fit(Xtr,ytr); pred=m.predict(Xte)
 rows.append([name,mean_absolute_error(yte,pred),mean_squared_error(yte,pred)**.5,r2_score(yte,pred)])
pd.DataFrame(rows,columns=["model","MAE","RMSE","R2"]).to_csv(out+"/model_comparison.csv",index=False)
best=models["Ridge"]; pred=best.predict(Xte)
plt.figure(figsize=(6,5)); plt.scatter(yte,pred,s=8,alpha=.4); plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Ridge: Actual vs Predicted"); plt.tight_layout(); plt.savefig(out+"/01_actual_vs_predicted.png",dpi=160); plt.close()
print(pd.DataFrame(rows,columns=["model","MAE","RMSE","R2"]).to_string(index=False))
