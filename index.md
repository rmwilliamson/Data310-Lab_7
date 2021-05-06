# Question 1

### If we use RandomForest (random_state=310) max_depth=10 and 1000 trees for ranking the importance of the input features the top three features are (in decreasing order)

```markdown
from sklearn.datasets import load_diabetes

columns ='age gender bmi map tc ldl hdl tch ltg glu'.split() 
diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns=columns)
y = diabetes.target

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=310, max_depth=10,n_estimators=1000)
df=pd.get_dummies(df)
model.fit(df,y)

features = df.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-9:]
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
```

## Answer =  'ltg', 'bmi', 'map'



# Question 2

### For the diabetes dataset you worked on the previous question, apply stepwise regression with add/drop p-values both set to 0.001. The model selected has the following input variables:

```markdown
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    
    included = list(initial_list)
    while True:
        changed=False

        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))


        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
    
result = stepwise_selection(df,y,[],0.001,0.001)
```

## Answer = ['bmi', 'ltg', 'map']


# Question 3

### For the diabetes dataset scale the input features by z-scores and then apply the ElasticNet model with alpha=0.1 and l1_ratio=0.5. If we rank the variables in the decreasing order of the absolute value of the coefficients the top three variables (in order) are

```markdown
from sklearn import linear_model as lm
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
Xs = scale.fit_transform(df)

model = lm.ElasticNet(alpha=0.1,l1_ratio = 0.5)
model.fit(Xs,y)
model.coef_

v = -np.sort(-np.abs(model.coef_))
for i in range(df.shape[1]):
  print(df.columns[np.abs(model.coef_)==v[i]])
 ```
 
 ## Answer = 'bmi', 'ltg', 'map'
 
 
 # Question 5
 
 ### In this problem consider 10-fold cross-validations and random_state=1693 for cross-validations and the decision tree. If you analyze the data with benign/malign tumors from breast cancer data with two features (radius_mean and texture_mean) and, according to what you learned about model selection, you try to determine the best maximum depth (in a range between 1 and 100) and the best minimum samples per leaf (in a range between 1 and 25) the optimal pair of hyper-parameters (such as max depth and min leaf samples) is
 
```markdown
df = pd.read_csv('drive/MyDrive/Colab Notebooks/MedicalData.csv')
y = df.diagnosis
X = df.loc[:,(df.columns != 'id') & (df.columns != 'diagnosis') & (df.columns != 'Unnamed: 32')].values

feats = ['radius_mean', 'texture_mean']
X = df[feats].values

from sklearn import tree

kf = KFold(n_splits=10, random_state=1693)
def DoKFold(X,y,model):
  accuracy = []
  for idxtrain, idxtest in kf.split(X):
    Xtrain = X[idxtrain,:]
    Xtest = X[idxtest,:]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    model.fit(Xtrain,ytrain)
    predicted_classes = model.predict(Xtest)
    accuracy.append(accuracy_score(ytest,predicted_classes))
  return (np.mean(accuracy))

model_one = tree.DecisionTreeClassifier(min_samples_leaf=19, max_depth=5, random_state=1693)
one = DoKFold(X,y,model_one)

model_two = tree.DecisionTreeClassifier(min_samples_leaf=4, max_depth=28, random_state=1693)
two = DoKFold(X,y,model_two)

model_three = tree.DecisionTreeClassifier(min_samples_leaf=9, max_depth=22, random_state=1693)
three = DoKFold(X,y,model_three)

model_four = tree.DecisionTreeClassifier(min_samples_leaf=4, max_depth=19, random_state=1693)
four = DoKFold(X,y,model_four)

model_four = tree.DecisionTreeClassifier(min_samples_leaf=4, max_depth=19, random_state=1693)
four = DoKFold(X,y,model_four)

model_five = tree.DecisionTreeClassifier(min_samples_leaf=3, max_depth=21, random_state=1693)
five = DoKFold(X,y,model_five)

print(one)
print(two)
print(three)
print(four)
print(five)
```
## Scores :
- 0.8892543859649124
- 0.8647243107769423
- 0.8699874686716791
- 0.8647243107769423
- 0.8577067669172932

## Answer = (19,5)


# Question 6

### In this problem consider 10-fold cross-validations and random_state=12345 for cross-validations and the decision tree. If you analyze the data with benign/malign tumors from breast cancer data with two features (radius_mean and texture_mean) and, according to what you learned about model selection, you try to determine the best maximum depth (in a range between 1 and 100) and the best minimum samples per leaf (in a range between 1 and 25) the number of False Negatives is:

```markdown
kf = KFold(n_splits=10, random_state=12345)
def DoKFold_6(X,y,model):
  accuracy = []
  for idxtrain, idxtest in kf.split(X):
    Xtrain = X[idxtrain,:]
    Xtest = X[idxtest,:]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    model.fit(Xtrain,ytrain)
    predicted_classes = model.predict(Xtest)
    accuracy.append(accuracy_score(ytest,predicted_classes))
  return (np.mean(accuracy))
  
model_one_6 = tree.DecisionTreeClassifier(min_samples_leaf=19, max_depth=5, random_state=12345)
one_6 = DoKFold_6(X,y,model_one_6)


spc = ['Malignant','Benign']
cm = CM(y,model_one_6.predict(X))
pd.DataFrame(cm, columns=spc, index=spc)
```

## Answer = 22


# Question 7

### In this problem consider 10-fold cross-validations and random_state=1693 for cross-validations and the decision tree. If you analyze the data with benign/malign tumors from breast cancer data set with two features (radius_mean and texture_mean) and, according to what you learned about model selection, you try to determine the best maximum depth (in a range between 1 and 100) and the best minimum samples per leaf (in a range between 1 and 25) the accuracy is about

```markdown
model_one = tree.DecisionTreeClassifier(min_samples_leaf=19, max_depth=5, random_state=1693)
one = DoKFold(X,y,model_one)
```

## Answer = 0.889


# Question 12

### In this problem the input features will be scaled by the z-scores and consider a use a random_state=1234. If you analyze the data with benign/malign tumors from breast cancer data, consider a decision tree with max_depth=10,min_samples_leaf=20 and fit on 9 principal components the number of true positives is:

```markdown
model = tree.DecisionTreeClassifier(max_depth=10,min_samples_leaf=20, random_state=1234)

df_12 = pd.read_csv('drive/MyDrive/Colab Notebooks/MedicalData.csv')
y = df.diagnosis
X_12 = df.loc[:,(df.columns != 'id') & (df.columns != 'diagnosis') & (df.columns != 'Unnamed: 32')].values

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
Xs = ss.fit_transform(X_12)

from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
y_dif = labelencoder_Y.fit_transform(y)

from sklearn.decomposition import PCA
pca = PCA(n_components=9)
principalComponents = pca.fit_transform(Xs)
principalDf = pd.DataFrame(data = principalComponents)
finalDf = pd.concat([principalDf, df[['diagnosis']]], axis = 1)

vals = finalDf.values
model.fit(finalDf,y)
predicted_classes_dtree = model.predict(vals)

spc = ['Malignant','Benign']
cm = CM(y,predicted_classes_dtree)
pd.DataFrame(cm, columns=spc, index=spc)
```


## Answer = 340
