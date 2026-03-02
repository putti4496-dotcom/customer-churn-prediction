import pandas as pd

data = pd.read_csv("Churn_Modelling.csv")

print(data.shape)
print(data.head())
print(data.columns)
import pandas as pd

data = pd.read_csv("Churn_Modelling.csv")

# Remove useless columns
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

print(data.head())
# Convert categorical columns
data = pd.get_dummies(data, drop_first=True)

print(data.head())
X = data.drop('Exited', axis=1)
y = data['Exited']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
