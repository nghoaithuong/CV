import numpy as np
import matplotlib.pyplot as plt
# height(cm)
x=np.array([147,150,153,155,158,160,163,165,168,170,173,175,178,180,183]) # 1x15
#weight(kg)
y=np.array([49,50,51,52,54,56,58,59,60,62,63,64,66,67,68])
# We have y=w_1*x + w_0 so need find w1 and w0
#print(x.shape[0])
print(y.shape)
print(x.shape[0])
x=x.reshape(-1,1) # 15x1
print(x.shape)
one = np.ones((x.shape[0],1), dtype = int) #15x1
print(one.shape)
xbar = np.concatenate((one,x), axis =1) # ghép 2 ma trận theo chiều hàng => 15x2
print(xbar.shape)
A= np.dot(xbar.T,xbar) # nhân ma trận: 2x15*15x2 = 2x2
B=np.dot(xbar.T,y) # nhân ma trận 2x15 * 15x1 = 2x1
w=np.dot(np.linalg.pinv(A),B) # Nghịch đảo ma trận A nhân B = 2x1
w0,w1=w[0],w[1]
print(A)
print(B)
print(w)
print(w0,w1)
#x1 = np.arange(11)
y1 = w1*x + w0
plt.plot(x, y1, x, y1, 'bx')
plt.plot(x,y,'ro')
plt.title('Compare')
plt.xlabel('height(cm)')
plt.ylabel('weight(kg)')
plt.show()

y2 = 155*w1 + w0
y3 = 160*w1 + w0
print(y2, y3)
