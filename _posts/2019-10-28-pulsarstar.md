---
title: "Pulsar Stars"
date: 2019-10-19
tags: [Prediction Models]
header:
  image:
excerpt: "Data Science, Imbalanced Data, Machine Learning"
classes: wide
---


# Predicting Pulsar Stars

We will be exploring supervised learning concepts by creating a model to predict the legitimacy of pulsar star candidates from the [HTRU](https://archive.ics.uci.edu/ml/datasets/HTRU2) dataset.

## Pulsar Stars

[Pulsars](https://en.wikipedia.org/wiki/Pulsar) are [Neutron stars](https://en.wikipedia.org/wiki/Neutron_star) that emit radiowaves detectable here on Earth. They rotate rapidly, emitting a periodic pattern accross the sky which can be detected by large radio telescopes. Each pulsar emits a different pattern due to variations in each rotation so each potential signal detection or candidate is averaged over many rotations determined by the length of the observation. The vast majority of these candidate signals end up actually being radio frequency interference and noise which makes finding a legitimate pulsar star through all the data difficult.

We can use machine learning tools to predict whether a candidate is legitimate pulsar star or illegitimate noise caused from interference by modeling candidate signals as a binary classification problem. The legitimate pulsar candidate signals are in the positive class and the illigitimate candidate signals are in the negative class.


## Understanding the Data

This data set contains 1,639 legitimate pulsar star candidates and 16,259 illegitimate candidate signals caused by RFI or noise.

Here is the attribute description directly from the [source](https://archive.ics.uci.edu/ml/datasets/HTRU2):
"
Each candidate is described by 8 continuous variables, and a single class variable. The first four are simple statistics obtained from the integrated pulse profile (folded profile). This is an array of continuous variables that describe a longitude-resolved version of the signal that has been averaged in both time and frequency. The remaining four variables are similarly obtained from the DM-SNR curve. These are summarised below:

1. Mean of the integrated profile.
2. Standard deviation of the integrated profile.
3. Excess kurtosis of the integrated profile.
4. Skewness of the integrated profile.
5. Mean of the DM-SNR curve.
6. Standard deviation of the DM-SNR curve.
7. Excess kurtosis of the DM-SNR curve.
8. Skewness of the DM-SNR curve.
9. Class

HTRU 2 Summary
17,898 total examples.
1,639 positive examples.
16,259 negative examples.
"

Let's read in the data.



```python
# import libraries
import numpy as np
import pandas as pd

# read in data
df = pd.read_csv('pulsar_stars.csv')

#display data shape and first few rows
print(df.shape)
df.head()
```

    (17898, 9)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean of the integrated profile</th>
      <th>Standard deviation of the integrated profile</th>
      <th>Excess kurtosis of the integrated profile</th>
      <th>Skewness of the integrated profile</th>
      <th>Mean of the DM-SNR curve</th>
      <th>Standard deviation of the DM-SNR curve</th>
      <th>Excess kurtosis of the DM-SNR curve</th>
      <th>Skewness of the DM-SNR curve</th>
      <th>target_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>140.562500</td>
      <td>55.683782</td>
      <td>-0.234571</td>
      <td>-0.699648</td>
      <td>3.199833</td>
      <td>19.110426</td>
      <td>7.975532</td>
      <td>74.242225</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>102.507812</td>
      <td>58.882430</td>
      <td>0.465318</td>
      <td>-0.515088</td>
      <td>1.677258</td>
      <td>14.860146</td>
      <td>10.576487</td>
      <td>127.393580</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>103.015625</td>
      <td>39.341649</td>
      <td>0.323328</td>
      <td>1.051164</td>
      <td>3.121237</td>
      <td>21.744669</td>
      <td>7.735822</td>
      <td>63.171909</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>136.750000</td>
      <td>57.178449</td>
      <td>-0.068415</td>
      <td>-0.636238</td>
      <td>3.642977</td>
      <td>20.959280</td>
      <td>6.896499</td>
      <td>53.593661</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>88.726562</td>
      <td>40.672225</td>
      <td>0.600866</td>
      <td>1.123492</td>
      <td>1.178930</td>
      <td>11.468720</td>
      <td>14.269573</td>
      <td>252.567306</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# display data types
df.dtypes
```




     Mean of the integrated profile                  float64
     Standard deviation of the integrated profile    float64
     Excess kurtosis of the integrated profile       float64
     Skewness of the integrated profile              float64
     Mean of the DM-SNR curve                        float64
     Standard deviation of the DM-SNR curve          float64
     Excess kurtosis of the DM-SNR curve             float64
     Skewness of the DM-SNR curve                    float64
    target_class                                       int64
    dtype: object




```python
#display Nan values
df.isna().sum()
```




     Mean of the integrated profile                  0
     Standard deviation of the integrated profile    0
     Excess kurtosis of the integrated profile       0
     Skewness of the integrated profile              0
     Mean of the DM-SNR curve                        0
     Standard deviation of the DM-SNR curve          0
     Excess kurtosis of the DM-SNR curve             0
     Skewness of the DM-SNR curve                    0
    target_class                                     0
    dtype: int64



All of the data is numeric and there is no missing data which is good. Let's replace the column names to make our lives easier.


```python
#format column names into snake_case format
col = ['mean_profile', 'std_profile', 'kurtosis_profile', 'skewness_profile',
 'mean_dmsnr_curve', 'std_dmsnr_curve', 'kurtosis_dmsnr_curve', 'skewness_dmsnr_curve', 'target_class']

df.columns = col
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_profile</th>
      <th>std_profile</th>
      <th>kurtosis_profile</th>
      <th>skewness_profile</th>
      <th>mean_dmsnr_curve</th>
      <th>std_dmsnr_curve</th>
      <th>kurtosis_dmsnr_curve</th>
      <th>skewness_dmsnr_curve</th>
      <th>target_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>140.562500</td>
      <td>55.683782</td>
      <td>-0.234571</td>
      <td>-0.699648</td>
      <td>3.199833</td>
      <td>19.110426</td>
      <td>7.975532</td>
      <td>74.242225</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>102.507812</td>
      <td>58.882430</td>
      <td>0.465318</td>
      <td>-0.515088</td>
      <td>1.677258</td>
      <td>14.860146</td>
      <td>10.576487</td>
      <td>127.393580</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>103.015625</td>
      <td>39.341649</td>
      <td>0.323328</td>
      <td>1.051164</td>
      <td>3.121237</td>
      <td>21.744669</td>
      <td>7.735822</td>
      <td>63.171909</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>136.750000</td>
      <td>57.178449</td>
      <td>-0.068415</td>
      <td>-0.636238</td>
      <td>3.642977</td>
      <td>20.959280</td>
      <td>6.896499</td>
      <td>53.593661</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>88.726562</td>
      <td>40.672225</td>
      <td>0.600866</td>
      <td>1.123492</td>
      <td>1.178930</td>
      <td>11.468720</td>
      <td>14.269573</td>
      <td>252.567306</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Data Visualization

Let's take a look at the class imbalance of the `target_class` variable which indicates whether or not the candidate is a legitimate pulsar star.


```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set(rc={'figure.figsize':(15,10)})
plt.pie(df["target_class"].value_counts().values,
        labels=["non-pulsar stars","pulsar stars"],
        autopct="%1.0f%%",wedgeprops={"linewidth":2,"edgecolor":"white"})
my_circ = plt.Circle((0,0),.7,color = "white")
plt.gca().add_artist(my_circ)
plt.subplots_adjust(wspace = .2)
plt.title("Proportion of target variable in dataset")
plt.show()
```


![png]({{ "/images/pulsar-star/output_7_0.png" }})


As noted, the `target_class` is severely imbalanced, out of 17,898 candidates, only 16,259 negative examples to only 1,639 positive examples. This is almost a 10 to 1 ratio.

We can use a heatmap to visualize the correlations between variables in order to see how each variable affects one another.


```python
sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Between Variables")
plt.show()
```


![png]({{ "/images/pulsar-star/output_9_0.png" }})


## Class Imbalance - Undersampling

This preliminary model produced a 97% accuracy. Due to the severity of the class imbalance we will have to employ additional techniques in order to produce a model that will not overfit. We want a model to can progress and evolve to many different data sets. In order to do this we will undersample the `target_class` variable meaning we will balance the amount of the positive cases to the amount of negative cases to create a 50/50 balance in the data set.


```python
# import sklearn libraries
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams

# catergorize dependent and independent features X and y
columns = df.columns.tolist()
columns = [c for c in columns if c not in ["target_class"]]
target = "target_class"
# define a random state
state = np.random.RandomState(13)
X = df[columns]
y = df[target]
# print the shapes of X and y
print(X.shape)
print(y.shape)
```

    (17898, 8)
    (17898,)



```python
# categorize pulsar and non-pulsars into 1 for positive and 0 for the negative class
pulsar = df[df['target_class']==1]
non_pulsar = df[df['target_class']==0]
```


```python
# print the number of positive classes and negative classes of the data
print(pulsar.shape,non_pulsar.shape)
```

    (1639, 9) (16259, 9)


### Undersampling

We see our positive class contains 1639 values compared to 16259 values in the negative class.


```python
from imblearn.under_sampling import NearMiss
```


```python
# undersample the data
nm = NearMiss(random_state=13)
X_res,y_res=nm.fit_sample(X,y)
```


```python
# print number of positve and negative classes
print(X_res.shape,y_res.shape)
```

    (3278, 8) (3278,)


Now we have created the variables `X_res` and `y_res` as a 50/50 balance to train our models.


```python
from collections import Counter
print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))
```

    Original dataset shape Counter({0: 16259, 1: 1639})
    Resampled dataset shape Counter({0: 1639, 1: 1639})


## Undersampling Models:

### Linear Regression

We will start with linear regression.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.5, random_state=13)
```


```python
# import sklearn tools
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# transform which scales between 1 and 0
sc = MinMaxScaler()
# instantiate model
lr = LogisticRegression(random_state=13)
# applies transformation on logistic regression model using a pipeline
pipe_lr = Pipeline([('scaler', sc), ('lr', lr)])
```


```python
# train model using the the transformed insantiated logistic regression model
pipe_lr.fit(X_train, y_train)
```

    /Users/Shaun/opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)





    Pipeline(memory=None,
             steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))),
                    ('lr',
                     LogisticRegression(C=1.0, class_weight=None, dual=False,
                                        fit_intercept=True, intercept_scaling=1,
                                        l1_ratio=None, max_iter=100,
                                        multi_class='warn', n_jobs=None,
                                        penalty='l2', random_state=13,
                                        solver='warn', tol=0.0001, verbose=0,
                                        warm_start=False))],
             verbose=False)




```python
# predict probabilites and keeps only positive outcomes
test_probas_lr = pipe_lr.predict_proba(X_test)[:,1]

# import error metric tools
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# generate a no skill prediction
ns_probs = [0 for _ in range(len(y_test))]

# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, test_probas_lr)

print('No Skill ROC-AUC score: %.2f' % ns_auc)
print('Linear Regression ROC-AUC score: %.2f' % lr_auc)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, test_probas_lr)

# figure size
plt.figure(figsize=(15,10))

# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill', color='blue')
plt.plot(lr_fpr, lr_tpr, linestyle='--', label="Logistic Regression", color='red')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
```

    No Skill ROC-AUC score: 0.50
    Linear Regression ROC-AUC score: 0.95



![png]({{ "/images/pulsar-star/output_24_1.png" }})


### Random Forest Classifier


```python
from sklearn.ensemble import RandomForestClassifier

#instantiate random forest classifier with a balanced class weight
rfc = RandomForestClassifier(class_weight='balanced', random_state=13)

pipe_rfc = Pipeline([('scaler', sc), ('rfc', rfc)])
```


```python
# train model using the the transformed insantiated random forest model
pipe_rfc.fit(X_train, y_train)
```

    /Users/Shaun/opt/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)





    Pipeline(memory=None,
             steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))),
                    ('rfc',
                     RandomForestClassifier(bootstrap=True, class_weight='balanced',
                                            criterion='gini', max_depth=None,
                                            max_features='auto',
                                            max_leaf_nodes=None,
                                            min_impurity_decrease=0.0,
                                            min_impurity_split=None,
                                            min_samples_leaf=1, min_samples_split=2,
                                            min_weight_fraction_leaf=0.0,
                                            n_estimators=10, n_jobs=None,
                                            oob_score=False, random_state=13,
                                            verbose=0, warm_start=False))],
             verbose=False)




```python
# predict probabilites and keep only positive outcomes
test_probas_rfc = pipe_rfc.predict_proba(X_test)[:,1]


# calculate scores
rfc_auc = roc_auc_score(y_test, test_probas_rfc)

print('No Skill ROC-AUC score: %.2f' % ns_auc)
print('Linear Regression ROC-AUC score: %.2f' % lr_auc)
print('Random Forest Classifier ROC-AUC score: %.2f' % rfc_auc)

# calculate roc curves
rfc_fpr, rfc_tpr, _ = roc_curve(y_test, test_probas_rfc)

# figure size
plt.figure(figsize=(15,10))

# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, linestyle='--', label="Logistic Regression", color='red')
plt.plot(rfc_fpr, rfc_tpr, linestyle='--', label="Random Forest Classifier", color='green')


# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
```

    No Skill ROC-AUC score: 0.50
    Linear Regression ROC-AUC score: 0.95
    Random Forest Classifier ROC-AUC score: 0.96



![png]({{ "/images/pulsar-star/output_28_1.png" }})


## Multinomial Naive Bayes


```python
from sklearn.naive_bayes import MultinomialNB

#instantiate multinomial naive bayes
nb = MultinomialNB()

pipe_nb = Pipeline([('scaler', sc), ('nb', nb)])
```


```python
# train model using the the transformed insantiated naive bayes model
pipe_nb.fit(X_train, y_train)
```




    Pipeline(memory=None,
             steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))),
                    ('nb',
                     MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))],
             verbose=False)




```python
# predict probabilites and keep only positive outcomes
test_probas_nb = pipe_nb.predict_proba(X_test)[:,1]

# calculate scores
nb_auc = roc_auc_score(y_test, test_probas_nb)

print('No Skill ROC-AUC score: %.2f' % ns_auc)
print('Linear Regression ROC-AUC score: %.2f' % lr_auc)
print('Random Forest Classifier ROC-AUC score: %.2f' % rfc_auc)
print('Multinomial Naive Bayes ROC-AUC score: %.2f' % nb_auc)

# calculate roc curves
nb_fpr, nb_tpr, _ = roc_curve(y_test, test_probas_nb)

# figure size
plt.figure(figsize=(15,10))

# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, linestyle='--', label="Logistic Regression", color='red')
plt.plot(rfc_fpr, rfc_tpr, linestyle='--', label="Random Forest Classifier", color='green')
plt.plot(nb_fpr, nb_tpr, linestyle='--', label='Naive Bayes', color='purple')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
```

    No Skill ROC-AUC score: 0.50
    Linear Regression ROC-AUC score: 0.95
    Random Forest Classifier ROC-AUC score: 0.96
    Multinomial Naive Bayes ROC-AUC score: 0.93



![png]({{ "/images/pulsar-star/output_32_1.png" }})



```python

```
