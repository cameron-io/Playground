## Intro

The Logistic Regression model doesn’t just return a prediction, but it returns a probability value between 0 and 1.

>= 0.5 == true
< 0.5 == false

If we make the threshold higher, we’ll have fewer positive predictions, but our positive predictions are more likely to be correct:
- Thus, precision will increase, recall will decrease.

## Sensitivity & Specificity

These values demonstrate the same trade-off that precision and recall demonstrate.

### Sensitivity:
- is another term for the recall, which is the true positive rate.

true positives / (true positives + false negatives)

### Specificity:
- is a new formula, the true negative rate.

true negatives / (true negatives + false positives)


### Example:

                    Actual      Actual
                    Positive    Negative
Predicted positive  TP: 30      FP: 20
Predicted negative  FN: 10      TN: 40

sensitivity = 30 / (30 + 10) = 0.75
specificity = 40 / (40 + 20) = 0.67

## Using Scikit-learn

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_recall_fscore_support

sensitivity_score = recall_score
def specificity_score(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    return r[0]

df = pd.read_csv('titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Fare']].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("sensitivity:", sensitivity_score(y_test, y_pred))
print("specificity:", specificity_score(y_test, y_pred))
```

When using scikit-learn’s predict method, you are given 0 and 1 values of the prediction.

However, behind the scenes the Logistic Regression model is getting a probability value between 0 and 1 for each datapoint and then rounding to either 0 or 1.

If we want to choose a different threshold besides 0.5, we’ll want those probability values.

```python
y_pred = model.predict_proba(X_test)[:, 1]
```

If comparing these probability values with our threshold
```python
y_pred = model.predict_proba(X_test)[:, 1] > 0.75
```