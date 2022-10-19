import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import scipy.special
from ann import ANN

# load data

train_file = open("./mnist_train_100.csv", 'r')
train_list = train_file.readlines()
train_file.close()

test_file = open("./mnist_test_10.csv", 'r')
test_list = test_file.readlines()
test_file.close()

# init model

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3

nn = ANN(input_nodes, hidden_nodes, output_nodes, learning_rate)

# train

epoch = 1
for i in range(epoch):
    for record in train_list:
        values = record.split(',')
        inputs = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
        labels = np.zeros(output_nodes) + 0.01
        labels[int(values[0])] = 0.99

        nn.train(inputs, labels)

# test

record = 0

# data
values = test_list[record].split(',')
print(values[0])
image = np.asfarray(values[1:]).reshape(28, 28)
plt.imshow(image, cmap='Greys', interpolation='None')
plt.show()

# ready
inputs = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01

# predict
outputs = nn.predict(inputs)
print("Prediction:")
print(outputs, end="\n\n")

# sort
indexs = np.argsort(-outputs, axis=0)
outputs_sum = sum(outputs)

for index in indexs:
    print(index, end=": ")
    print("%.2f" % (outputs[index]/outputs_sum*100), end="%\n")
