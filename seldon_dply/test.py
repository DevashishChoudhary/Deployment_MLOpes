import pandas as pd 
import requests 

csv_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
df = pd.read_csv(csv_url, sep=";")

# choose features
features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"] 

feature_df = df[features]
row = list(feature_df.loc[1,:])

r = requests.post("http://localhost:5001/predict", json={"data":{"ndarray":[row]}})
print(r.content)
