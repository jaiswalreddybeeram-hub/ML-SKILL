
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
df=pd.read_csv("titanic.csv")
df=df.drop_duplicates()
target="Survived"
features=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
X=df[features]; y=df[target]
num=["Pclass","Age","SibSp","Parch","Fare"]; cat=["Sex","Embarked"]
pre=ColumnTransformer([
 ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),num),
 ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)
])
Xt=pre.fit_transform(X)
names=pre.get_feature_names_out()
clean=pd.DataFrame(Xt,columns=names)
clean["Survived"]=y.reset_index(drop=True)
clean.to_csv("titanic_cleaned.csv",index=False)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
pd.DataFrame({"train_rows":[len(X_train)],"test_rows":[len(X_test)],"features_after_encoding":[Xt.shape[1]]}).to_csv("outputs/preprocessing_summary.csv",index=False)
print("Cleaned dataset saved as titanic_cleaned.csv")
print("Numeric: median imputation + standard scaling; categorical: mode imputation + one-hot encoding.")
