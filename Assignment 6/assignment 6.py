import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.naive_bayes import GaussianNB


data = pd.read_csv('tennis.csv')
print(data.head())

X = data.iloc[:,1:-1]
y = data.iloc[:,-1]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=30)


# Build a Gaussian Classifier
model = GaussianNB()

# Model training
model.fit(X_train, y_train)

# Predict Output
predicted = model.predict(X_test)

print("\nActual Value:\n", y_test)
print("\nPredicted Value:\n", predicted)


y_pred = model.predict(X_test)
accuray = accuracy_score(y_pred, y_test)
precision = precision_score(y_test, y_pred, pos_label='yes')


print("\nAccuracy:", accuray)
print("\nPrecision:", precision)
