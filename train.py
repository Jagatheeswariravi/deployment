import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"

names =["preg","plas","pres","skin","test","mass","predi","age","class"]


df = pd.read_csv(url,names=names)

print(df)

array = df.values

X= array[:,0:8]

y= array[:,8]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)

model = LogisticRegression()

model.fit(X_train,y_train)

result = model.score(X_test,y_test)

print(result)

joblib.dump(model,open("diabetes.pkl","wb")