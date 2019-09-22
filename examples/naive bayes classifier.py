from __future__ import print_function
import numpy as np
from sklearn.naive_bayes import MultinomialNB
#train data
d1=[2,1,1,0,0,0,0,0,0]
d2=[1,1,0,1,1,0,0,0,0]
d3=[0,1,0,0,1,1,0,0,0]
d4=[0,1,0,0,0,0,1,1,1]
train_data = np.array([d1,d2,d3,d4])
print(train_data)
label = np.array(['B','B','B','N'])
#test_data
d5=np.array([[2,0,0,1,0,0,0,1,0]])
d6=np.array([[0,1,0,0,0,0,0,1,1]])
#call multinomialNB
model = MultinomialNB()
#training
model.fit(train_data,label)
print ('class d5:', str(model.predict(d5)[0]))
print('xác suất d5 vào mỗi lớp', model.predict_proba(d5))

print ('class d6:', str(model.predict(d6)[0]))
print('xác suất d6 vào mỗi lớp', model.predict_proba(d6))
