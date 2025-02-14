# Computing Accuracy

## Basic:
- equation: correct_predications/total
- Accuracy is a good measure if our classes are evenly split
- Very misleading if we have imbalanced classes.
```
100 datapoints, predicted 70 correctly, 30 incorrectly = 70%.
```
## Confusion Matrix:
- equation: (actual pos + actual neg)/(total)
```
                    Actual      Actual
                    Positive    Negative
Predicted positive  TP: 233     FP: 65
Predicted negative  FN: 109     TN: 480
* NOTE: Scikit-learn shows the negative counts first!

Accuracy:
(233+480)/(233+65+109+480) = 713/887 = 80.38%
```

### Commonly used metrics for precision and recall

1. Precision:
    - percentage of positive results which are relevant.
    - equation:
        ```
        True Positives / (True Positives + False Positives)
        ```

Using the Confusion Matrix example:
```
precision = 233 / (233 + 65) = 0.7819
```

2. Recall:
    - percentage of positive cases correctly classified.
    - equation:
        ```
        True Positives / (True Positives + False Negatives)
                                                 ^-- difference
        ```
Using the Confusion Matrix example:
```
precision = 233 / (233 + 109) = 0.6813
```

#### Trade-offs between
- increasing the recall (while lowering the precision)
- increasing the precision (and lowering the recall)

The higher the false-positives, the lower the precision.

if (false-positives will cause severe issues in the scenario being applied):
- increase precision, lower re-call.
else:
- balance both.

#### F1 Score
- Averages Precision & Recall
- equation:
    ```
    2 * (precision * recall) / (precision + recall)
    ```

Using the Confusion Matrix example:
```
f1 = 2 (0.7819) (0.6813) / (0.7819 + 0.6813) 
f1 = 0.7281
```

## Using Metrics in Scikit-learn

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv('titanic.csv')
df['male'] = df['Sex'] == 'male'

X = df[['Pclass', 'male', 'Age', 'Fare']].values
y = df['Survived'].values

model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)

print("accuracy:", accuracy_score(y, y_pred))
print("precision:", precision_score(y, y_pred))
print("recall:", recall_score(y, y_pred))
print("f1 score:", f1_score(y, y_pred))
print("confusion_matrix:", confusion_matrix(y, y_pred))
```
