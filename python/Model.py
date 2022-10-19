import numpy as np
import scipy.special


# 求矩阵各行平均值
def mavg(x):
    y = np.mean(x, axis = 1)
    y.shape = (y.size, 1)
    return y


class Activate:

    def __init__(self):
        pass


    def __call__(self, x):
        return None
    

    def d(self, x):
        return None


class Sigmoid(Activate):

    def __call__(self, x):
        sx = scipy.special.expit(x)
        self.t = sx
        return sx
    
    def d(self, x):
        sx = self.t
        return sx * (1 - sx)



class Layer:

    def __init__(self):
        self.next = None


    # 输入xs，返回ys，并缓存
    def forwards(self, inputs):
        return None

    
    # 输入y，返回x
    def backward(self, output):
        return None

    
    # 输入e，利用缓存修正w
    def train(self, error):
        pass


class Linear(Layer):

    def __init__(self, input, output, lr, activate = None):
        # (o*n) = (o*i) * (i*n) + (o*n)
        self.w = np.random.normal(0.0, pow(output, -0.5), (output, input))
        self.b = np.random.normal(0.0, pow(output, -0.5), (output, 1))
        self.a:Activate = activate
        self.lr = lr
        self.prev:Layer = None
        self.next:Layer = None
    

    def __call__(self, x:Layer):
        x.next = self
        self.prev = x
        return self
    

    def forwards(self, xs):
        # y = a(wx + b)
        self.cache_x = mavg(xs)
        ys = np.dot(self.w, xs) + self.b # b will auto reshape (o*1) to (o*n)
        self.cache_wxb = mavg(ys)
        if self.a:
            ys = self.a(ys)
        return ys


    def backward(self, y):
        # x = w * (y-b)
        x = np.dot(self.w.T, (y - self.b))
        return x

    
    def train(self, e):
        if self.a:
            # a'(wx) * x
            dydw = np.dot(self.a.d(self.cache_wxb), self.cache_x.T)
            dydb = self.a.d(self.cache_wxb)
        else:
            dydw = self.cache_x
            dydb = 1
        # e = y_ - y
        # w += lr * e * dy/dw
        self.w += self.lr * e * dydw
        self.b += self.lr * 0.001 * e * dydb


def Train(l:Layer, xs, ys_):
    ys = l.forwards(xs)
    while l.next:
        l = l.next
        ys = l.forwards(ys)
    
    e = mavg(ys_ - ys)
    
    pe = l.backward(e)
    l.train(e)
    while l.prev:
        l = l.prev
        e = pe
        pe = l.backward(e)
        l.train(e)


def Predict(l:Layer, xs):
    ys = l.forwards(xs)
    while l.next:
        l = l.next
        ys = l.forwards(ys)
    
    return ys
