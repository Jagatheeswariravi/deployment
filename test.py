import joblib
model = joblib.load("diabetes.pkl")

result = model.predict([[4,110,92,0,0,37.6,0.191,30]])

if result == 0:

    print("not diabetic")
else:
    print("is diabetic")
