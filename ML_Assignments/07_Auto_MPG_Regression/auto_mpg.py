
import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
df=pd.read_csv("auto_mpg.csv")
X=df.drop(columns="mpg"); y=df.mpg
num=["cylinders","displacement","horsepower","weight","acceleration","model_year"]; cat=["origin"]
pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),
                       ("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
model=Pipeline([("pre",pre),("reg",LinearRegression())])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
model.fit(Xtr,ytr); pred=model.predict(Xte)
m={"MAE":mean_absolute_error(yte,pred),"RMSE":mean_squared_error(yte,pred)**.5,"R2":r2_score(yte,pred)}
pd.DataFrame([m]).to_csv("outputs/metrics.csv",index=False)
plt.figure(figsize=(6,5)); plt.scatter(yte,pred,s=10,alpha=.5); plt.xlabel("Actual MPG"); plt.ylabel("Predicted MPG"); plt.title("Auto MPG Regression"); plt.tight_layout(); plt.savefig("outputs/actual_vs_predicted.png",dpi=160); plt.close()
print(pd.DataFrame([m]))
