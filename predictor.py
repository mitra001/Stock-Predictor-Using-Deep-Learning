'''

This script shows how to predict stock prices using a basic RNN

'''

import torch

import torch.nn as nn

from torch.autograd import Variable

import numpy as np

import os

import matplotlib
matplotlib.use('TkAgg')


torch.manual_seed(777)  # reproducibility



if "DISPLAY" not in os.environ:

    # remove Travis CI Error

    matplotlib.use('TkAgg')



import matplotlib.pyplot as plt





def MinMaxScaler(data):

    ''' Min Max Normalization



    Parameters

    ----------

    data : numpy.ndarray

        input data to be normalized

        shape: [Batch size, dimension]



    Returns

    ----------

    data : numpy.ndarry

        normalized data

        shape: [Batch size, dimension]



    References

    ----------

    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html



    '''

    numerator = data - np.min(data, 0)

    denominator = np.max(data, 0) - np.min(data, 0)

    # noise term prevents the zero division

    return numerator / (denominator + 1e-7)





# train Parameters

learning_rate = 0.01

num_epochs = 500

input_size = 6    # ¿if we change the imput size, can we add more parameters for the guess by simply adding one collumn to the csv file ?
                ############################################################################################
hidden_size = 5 # ¿We are cofused about hidden_size and num_layers? ¿is hidden size num of hidden layers?

num_classes = 1
                            #####################################################################################
timesteps = seq_length = 7 # ¿Does tis min the number of h, but what is x in each timestep? Go to the forth loop.

num_layers = 1  # number of layers in RNN



# Open, High, Low, Volume, Close

xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')

xy = xy[::-1]  # reverse order (chronically ordered)

xy = MinMaxScaler(xy)

x = xy

y = xy[:, [-1]]  # Close as label



# build a dataset

dataX = []

dataY = []

for i in range(0, len(y) - seq_length):      #########################################################################
                                             ## What is the relation between timestampe and the way the data is structured?
    _x = x[i:i + seq_length]

    print(_x.shape)

    _y = y[i + seq_length]  # Next close price

    print(_x, "->", _y)
 
    dataX.append(_x)

    dataY.append(_y)



# train/test split
                                    #############################################################################
train_size = int(len(dataY) * 0.7)   #¿why is the train size multiplyed by 0.7? ¿What does this means?

test_size = len(dataY) - train_size

trainX = torch.Tensor(np.array(dataX[0:train_size]))

trainX = Variable(trainX)

testX = torch.Tensor(np.array(dataX[train_size:len(dataX)]))

testX = Variable(testX)

trainY = torch.Tensor(np.array(dataY[0:train_size]))

trainY = Variable(trainY)

testY = torch.Tensor(np.array(dataY[train_size:len(dataY)]))

testY = Variable(testY)





class LSTM(nn.Module):



    def __init__(self, num_classes, input_size, hidden_size, num_layers):

        super(LSTM, self).__init__()

        self.num_classes = num_classes

        self.num_layers = num_layers

        self.input_size = input_size

        self.hidden_size = hidden_size

        self.seq_length = seq_length

        # Set parameters for RNN block

        # Note: batch_first=False by default.

        # When true, inputs are (batch_size, sequence_length, input_dimension)

        # instead of (sequence_length, batch_size, input_dimension)

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,

                            num_layers=num_layers, batch_first=True)

        # Fully connected layer

        self.fc = nn.Linear(hidden_size, num_classes)



    def forward(self, x):

        # Initialize hidden and cell states

        h_0 = Variable(torch.zeros(

            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(

            self.num_layers, x.size(0), self.hidden_size))



        # Propagate input through LSTM

        _, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out





# Instantiate RNN model

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)



# Set loss and optimizer function
                                  ####################################################################################
criterion = torch.nn.MSELoss()    # mean-squared error for regression, loss function   ¿what else apart from mse?

optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)  # optimization algorithm  ¿what else apart from this?

# try: cross entropy loss with SGD

# Train the model

for epoch in range(num_epochs):

    outputs = lstm(trainX)

    optimizer.zero_grad()

    # obtain the loss function

    loss = criterion(outputs, trainY)
                                   ################################################################
    loss.backward()                ## ¿Does this means backward propagation? ¿If we change for foward is going to do foward propagation?

    optimizer.step()

    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))



print("Learning finished!")



# Test the model

lstm.eval()

test_predict = lstm(testX)



# Plot predictions

test_predict = test_predict.data.numpy()

testY = testY.data.numpy()

plt.plot(testY)

plt.plot(test_predict)

plt.xlabel("Time Period")

plt.ylabel("Stock Price")

plt.savefig("demo.png")

plt.show()