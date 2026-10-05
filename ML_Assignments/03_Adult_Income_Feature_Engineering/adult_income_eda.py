import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
df=pd.read_csv('adult.csv'); out=Path('outputs'); out.mkdir(exist_ok=True)
df['workclass']=df['workclass'].fillna('Unknown'); df['occupation']=df['occupation'].fillna('Unknown'); df['capital_gain']=df['capital_gain'].fillna(df['capital_gain'].median())
df['age_group']=pd.cut(df['age'],[16,25,35,45,55,65,100],labels=['17-25','26-35','36-45','46-55','56-65','66+'])
df['is_high_income']=(df['income']=='>50K').astype(int); df['hours_category']=pd.cut(df['hours_per_week'],[0,30,40,50,100],labels=['Part-time','Full-time','Overtime','Extreme']); df['capital_gain_log']=np.log1p(df['capital_gain'])
df.to_csv('adult_engineered.csv',index=False); df.describe(include='all').transpose().to_csv(out/'descriptive_statistics.csv')
plt.figure(figsize=(7,4)); df['income'].value_counts().plot(kind='bar'); plt.title('Income Class Distribution'); plt.tight_layout(); plt.savefig(out/'01_income_distribution.png',dpi=160); plt.close()
rate=df.groupby('education')['is_high_income'].mean().sort_values(ascending=False); plt.figure(figsize=(8,4)); rate.plot(kind='bar'); plt.xticks(rotation=35,ha='right'); plt.title('High-Income Rate by Education'); plt.tight_layout(); plt.savefig(out/'02_income_by_education.png',dpi=160); plt.close()
plt.figure(figsize=(7,4)); df.boxplot(column='hours_per_week',by='income'); plt.suptitle(''); plt.title('Hours per Week by Income'); plt.tight_layout(); plt.savefig(out/'03_hours_by_income.png',dpi=160); plt.close()
print('Engineered features: age_group, is_high_income, hours_category, capital_gain_log.')
