import pandas
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data=pd.read_csv('F:\Muzammil\Assignmnet 1\Country.csv',na_values=["?"])
print("Data: \n",data)

print("---------------------------------------")
print("\nMissing Data:")
print(data.isnull().sum())

print("---------------------------------------")
print("\nFilling Nan values with 0:")
df=data.fillna(value=0)
print(df)

data.describe()
data['Age'].fillna(data['Age'].mean(),inplace=True)
data.isnull().sum()
data['Income'].fillna(data['Income'].median(),inplace=True)
data.isnull().sum()

print("---------------------------------------")
print("Scaling")
scale = StandardScaler()
df = pd.read_csv('F:\Muzammil\Assignmnet 1\Country.csv',na_values=["?"])
X = df[['Age','Income']]
scaledX = scale.fit_transform(X)
print(scaledX)

print("---------------------------------------")
print("Categorial Data")
label = LabelEncoder()
df = label.fit_transform(df['Region'])
print(df)

dataset=pd.read_csv('F:\Muzammil\Assignmnet 1\Country.csv')
print("---------------------------------------")
print("\nDataset Preview")
print(dataset)

X = dataset[['Region','Age','Income']]
y = dataset['Online Shopper']

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

print("---------------------------------------")
print("\nTrain(80%):")
print(pd.concat([X_train,y_train],axis=1))

print("---------------------------------------")
print("\nTest(20%):")
print(pd.concat([X_test,y_test],axis=1))

