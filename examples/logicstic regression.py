from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)
#train data
x=np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.00]])
y=np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

print(x.shape[1])
print(y.shape)
#xbar=np.ones(x, x.shape[1])
x= np.concatenate((np.ones((1, x.shape[1])), x), axis =0)

# hàm sigmoid trong logicstic
def sigmoid(s):
    return 1/(1 + np.exp(-s))

# hàm logicstic 
def logistic_sigmoid_regression(x, y, w_init, eta, tol = 1e-4, max_count = 10000):
    w = [w_init]
    it = 0
    N = x.shape[1]
    d = x.shape[0]
    count = 0
    check_w_after = 20
    while count < max_count:
        # mix data
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = x[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta*(yi - zi)*xi
            count += 1
            # stopping criteria
            if count%check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w
eta = .05
d = x.shape[0]
w_init = np.random.randn(d, 1)

w = logistic_sigmoid_regression(x, y, w_init, eta)
print(w[-1])

print(sigmoid(np.dot(w[-1].T, x)))
# biểu diễn trên đồ thị

x0 = x[1, np.where(y == 0)][0]
y0 = y[np.where(y == 0)]
x1 = x[1, np.where(y == 1)][0]
y1 = y[np.where(y == 1)]

plt.plot(x0, y0, 'ro', markersize = 8)
plt.plot(x1, y1, 'bs', markersize = 8)

xx = np.linspace(0, 6, 1000)
w0 = w[-1][0][0]
w1 = w[-1][1][0]
threshold = -w0/w1
yy = sigmoid(w0 + w1*xx)
plt.axis([-2, 8, -1, 2])
plt.plot(xx, yy, 'g-', linewidth = 2)
plt.plot(threshold, .5, 'y^', markersize = 8)
plt.xlabel('studying hours')
plt.ylabel('predicted probability of pass')
plt.show()
