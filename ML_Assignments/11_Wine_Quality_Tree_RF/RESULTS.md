# Results

### confusion_matrix.csv

,Pred Low,Pred Medium,Pred High
Actual Low,272,5,0
Actual Medium,33,10,0
Actual High,0,0,0


### feature_importance.csv

,0
alcohol,0.3830717509127201
volatile_acidity,0.09288638075524869
fixed_acidity,0.0768027602708332
pH,0.06398228417399539
sulphates,0.06057802006437182
density,0.059746272913279753
total_sulfur_dioxide,0.058252912620655725
free_sulfur_dioxide,0.056252136449888815
residual_sugar,0.05186853775799979
citric_acid,0.0502803039940955
chlorides,0.046278640086911293


### model_comparison.csv

model,accuracy,precision_weighted,recall_weighted,f1_weighted
DecisionTree,0.853125,0.8275434231786134,0.853125,0.8353422973187371
RandomForest,0.88125,0.8615505464480874,0.88125,0.8554427361061737
