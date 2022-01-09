#library
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

#dataset 
df=pd.read_csv('/content/batik.csv', sep=",")

x = np.array(df.iloc[:,1]).reshape((-1, 1))

y = np.array(df.iloc[:,0])

#call model regression
model = LinearRegression().fit(X,Y)

#save model
file = 'model.sav'
joblib.dump(model, file)
