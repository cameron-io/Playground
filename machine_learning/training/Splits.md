## Training and Test Sets

Splitting the dataset into a single training set and test set for evaluation purposes might yield an inaccurate measure of the evaluation metrics when the dataset is small.

### Using Multiple Training and Test Sets

Instead of just taking a chunk of the data as the test set, let’s break our dataset into 5 chunks. Let’s assume we have 200 datapoints in our dataset.

Each of these 5 chunks will serve as a test set. When Chunk 1 is the test set, we use the remaining 4 chunks as the training set. 

Thus we have 5 training and test sets as follows:

                                            Accuracy
1   Train   Train   Train   Train   *Test*  0.83
2   Train   Train   Train   *Test*  Train   0.79
3   Train   Train   *Test*  Train   Train   0.78
4   Train   *Test*  Train   Train   Train   0.80
5   *Test*  Train   Train   Train   Train   0.75

Each of the 5 times we have a test set of 20% (40 datapoints) and a training set of 80% (160 datapoints).

Every datapoint is in exactly 1 test set.

In this example, the reported Accuracy is the mean of 5 values:
= (0.83+0.79+0.78+0.80+0.75)/5 = 0.79

This process is called k-fold cross validation. The k is the number of chunks we split our dataset into. The standard number is 5, as we did in our example above.

The goal in cross validation is to get accurate measures for our metrics (accuracy, precision, recall):
- thus, we are building extra models in order to feel confident in the numbers we calculate and report.

In this example, each of the 5 models were built just for evaluation purposes, to report the metric values:
- the best possible model is going to be a model that uses all of the data


## Using Scikit-learn

```python
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

df = pd.read_csv('titanic.csv')
df['male'] = df['Sex'] == 'male'

X = df[['Pclass', 'male', 'Age', 'Fare']].values
y = df['Survived'].values

scores = []
kf = KFold(n_splits=5, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))

print(scores)
print(np.mean(scores))

final_model = LogisticRegression()
final_model.fit(X, y)
```

The final single precision value is represented by
```python
np.mean(scores)
```