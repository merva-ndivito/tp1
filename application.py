import pandas as pd
import joblib

test=[[20,1,4]]

model=joblib.load('model.pkl')

print(model.predict(test)[0])