import numpy as np
from sklearn import datasets, linear_model
# height(cm)
x=np.array([[147,150,153,155,158,160,163,165,168,170,173,175,178,180,183]]).T # 1x15
#weight(kg)
y=np.array([49,50,51,52,54,56,58,59,60,62,63,64,66,67,68])
# fit the model by linear regression
regr = linear_model.LinearRegression()
regr.fit(x,y) # in scikit-learn earch sample is one row
print('results:', regr.coef_[0],regr.intercept_)
