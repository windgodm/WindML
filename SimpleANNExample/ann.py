import numpy as np
import scipy.special

class ANN:
    """
    Artificial Neural Networks
    """
    
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):

        self.iN = inputNodes
        self.hN = hiddenNodes
        self.oN = outputNodes
        self.lr = learningRate

        self.wih = np.random.normal(0.0, pow(self.hN, -0.5), (self.hN, self.iN))
        self.who = np.random.normal(0.0, pow(self.oN, -0.5), (self.oN, self.hN))

        self.a_f = lambda x: scipy.special.expit(x)


    def predict(self, inputs_list):

        #translation
        i = np.array(inputs_list, ndmin=2).T

        # hidden
        h_i = np.dot(self.wih, i)
        h_o = self.a_f(h_i)

        # output
        o_i = np.dot(self.who, h_o)
        o_o = self.a_f(o_i)

        return o_o


    def train(self, inputs_list, outputs_list):

        # translation
        i = np.array(inputs_list, ndmin=2).T
        t = np.array(outputs_list, ndmin=2).T

        # hidden
        h_i = np.dot(self.wih, i)
        h_o = self.a_f(h_i)

        # output
        o_i = np.dot(self.who, h_o)
        o_o = self.a_f(o_i)

        # error
        o_e = t - o_o
        h_e = np.dot(self.who.T, o_e)

        # fix
        #r * ( (E*o*(1-o)) * lo.T)
        self.who += self.lr * np.dot(o_e * o_o * (1 - o_o), np.transpose(h_o))

        self.wih += self.lr * np.dot(h_e * h_o * (1 - h_o), np.transpose(i))
