from google.colab import files
uploaded=files.upload()

import pandas as pd
import numpy as np
data=pd.read_csv("Iris (1).csv")

data.describe()

data.head()

This will give you the number of rows and columns in the dataset.



data.shape

This will give you the data type of each column in the dataset, such as integer, float, or object.




data.dtypes

This will give you the unique values of the 'species' column, which should be setosa, versicolor, and virginica in the case of the Iris dataset.



data["Species"].unique()

for visualize


import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(data,hue="Species")
plt.show()

plt.boxplot(data["PetalLengthCm"])
plt.show()

plt.boxplot(data["SepalLengthCm"])
plt.show()

plt.boxplot(data["PetalWidthCm"])
plt.show()

plt.boxplot(data["SepalLengthCm"])
plt.show()


from sklearn.linear_model import LinearRegression
x=data.iloc[:,0:4]
y=data.iloc[:,4]
x
y


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)


model=LinearRegression()
model.fit(x,y)

model.score(x,y)

model.coef_

model.intercept_

y_pred=model.predict(x_test)


print("Mean Squared Error: %.2f" % np.mean((y_pred - y_test)**2))




THANK U
