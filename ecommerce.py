import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

customers = pd.read_csv("Ecommerce Customers.csv")

#customers.head()
#customers.info()
#sns.pairplot(customers)
#plt.show()
#print(customers.describe())
#print(customers.columns)

#sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)
#sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)
#sns.jointplot(x = 'Time on App', y = 'Length of Membership', kind = 'hex', data=customers)
#sns.pairplot(customers)



#most correlated feature with Yearly Amount Spent is Length of Membership

#sns.lmplot(x='Length of Membership', y = 'Yearly Amount Spent', data=customers)


#by analysing the data we get that the more the length of membership the more we get the yearly amount spent which usually makes sense in real world

#training and testing the data now

X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

lm = LinearRegression()
lm.fit(X_train, y_train)

print(lm.coef_)
#The coefficients tell you how much the target variable is expected to increase (or decrease) with a one-unit increase in the corresponding feature, holding all other features constant.

predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y values')


e1 = metrics.mean_absolute_error(y_test, predictions)
e2 = metrics.mean_squared_error(y_test, predictions)
e3 = np.sqrt(metrics.mean_squared_error(y_test, predictions))
print('MAE',e1,'\n','MSE', e2,'\n''RMSE',e3)

sns.displot((y_test - predictions),bins=50)
plt.show()

cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
print(cdf)

#by the cdf values we get 2 posibilities here. one is here that the company should focus more on the mobile app as it has the more timescreen as compared to the website one whereas the other solution could be that we should develop the website better so it catches with the mobile application. this both answers basically depends on the performance of the company and where the most users are usually active. in simple words we need more data to help determine the best results for it.
 