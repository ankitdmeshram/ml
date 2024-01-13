import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn import metrics

data = pd.read_csv('email.csv')
print(data.head())
data['Prediction'] = data['Prediction'].replace(0,"spam") 
data['Prediction'] = data['Prediction'].replace(1,"ham")

X = data.iloc[:,1:-1]
y = data.iloc[:,-1]
print(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

 
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam')
print("KNN Accuracy:", accuracy)
print("KNN Precision:",precision)


svm = SVC (kernel = 'linear',random_state = 0) 
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

 
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("SVM Precision:",metrics.precision_score(y_test, y_pred, pos_label='spam'))
