import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
df=pd.read_csv('titanic.csv'); out=Path('outputs'); out.mkdir(exist_ok=True)
pd.DataFrame({'rows':[len(df)],'columns':[df.shape[1]],'missing_cells':[int(df.isna().sum().sum())],'duplicates':[int(df.duplicated().sum())],'survival_rate':[df.Survived.mean()]}).to_csv(out/'dataset_summary.csv',index=False)
df.describe(include='all').transpose().to_csv(out/'descriptive_statistics.csv')
plt.figure(figsize=(6,4)); df['Survived'].value_counts().sort_index().plot(kind='bar'); plt.title('Titanic Survival Distribution'); plt.xlabel('Survived'); plt.ylabel('Count'); plt.tight_layout(); plt.savefig(out/'01_survival_distribution.png',dpi=160); plt.close()
plt.figure(figsize=(6,4)); df.groupby('Sex',dropna=False)['Survived'].mean().plot(kind='bar'); plt.title('Survival Rate by Sex'); plt.ylabel('Survival Rate'); plt.tight_layout(); plt.savefig(out/'02_survival_by_sex.png',dpi=160); plt.close()
plt.figure(figsize=(6,4)); df.groupby('Pclass')['Survived'].mean().plot(kind='bar'); plt.title('Survival Rate by Passenger Class'); plt.ylabel('Survival Rate'); plt.tight_layout(); plt.savefig(out/'03_survival_by_class.png',dpi=160); plt.close()
plt.figure(figsize=(7,4));
for val in [0,1]: df.loc[df.Survived==val,'Age'].dropna().plot(kind='hist',bins=25,alpha=.5,label=f'Survived={val}')
plt.legend(); plt.title('Age Distribution by Survival'); plt.xlabel('Age'); plt.tight_layout(); plt.savefig(out/'04_age_survival.png',dpi=160); plt.close()
df.groupby('Pclass')['Survived'].mean().to_csv(out/'survival_rate_by_class.csv')
print('ML lifecycle: Problem -> Data collection -> Cleaning -> EDA -> Feature preparation -> Modeling -> Evaluation -> Deployment/monitoring')
