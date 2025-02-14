## Overfitting
- Caused by feeding same data to model during training
- Predictions become:
    - Really good with seen data.
    - Poor with new data

## Dataset Types

Training Set:
- Used for building model

Test Set:
- Use to evaluate the model

Data Split:
- Training Set: 70-80%
- Test Set: 20-30%

## Using scikit-learn

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('titanic.csv')
df['male'] = df['Sex'] == 'male'

X = df[['Pclass', 'male', 'Age', 'Fare']].values
y = df['Survived'].values

# scikit-learn defaults to 75% / 25% split
# splits randomly, unless supplied random_state with arbitrary value
X_train, X_test, y_train, y_test = train_test_split(X, y)

print("whole dataset:", X.shape, y.shape)
print("training set:", X_train.shape, y_train.shape)
print("test set:", X_test.shape, y_test.shape)
```
Output:
```
whole dataset: (887, 6) (887,)
training set: (665, 6) (665,)
test set: (222, 6) (222,)
```

Applying a Logistic Regression model
```python
# building
model = LogisticRegression()
model.fit(X_train, y_train)

# evaluating
y_pred = model.predict(X_test)

print("accuracy:", accuracy_score(y_test, y_pred))
print("precision:", precision_score(y_test, y_pred))
print("recall:", recall_score(y_test, y_pred))
print("f1 score:", f1_score(y_test, y_pred))
```

If (accuracy, precision, recall, F1 score) very similar to the values when we used the entire dataset:
- This is a sign our model is not overfit.
