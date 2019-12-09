import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import sys
import requests
from torch.autograd import Variable

apikey = 'IVQTTYEFMDLY30AX'
global_filename = ''
input_size_g = 5
company_name_g = ''
indicator_g = ''


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
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


def MinMaxUnscaler(values, max, min):
    prices = values * (max - min)
    prices += min

    return prices


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
    '''

    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


def predictor():
    matplotlib.use('TkAgg')
    torch.manual_seed(777)  # reproducibility
    global input_size_g
    if "DISPLAY" not in os.environ:
        # remove Travis CI Error
        matplotlib.use('TkAgg')

    learning_rate = 0.01
    num_epochs = 500
    input_size = input_size_g  
    print(input_size)
    hidden_size = 5  
    num_classes = 1
    timesteps = seq_length = 7  
    num_layers = 1  

    # Open, High, Low, Volume, Close

    # xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
    xy = np.loadtxt(global_filename, delimiter=',')
    xy = xy[::-1]  # reverse order (chronically ordered)
    close = xy[:, [-1]] # close as label
    maxValue = max(close)
    minValue = min(close)
    xy = MinMaxScaler(xy)
    x = xy
    y = xy[:, [-1]]  # Close as label

    # build a dataset
    dataX = []
    dataY = []

    for i in range(0, len(y) - seq_length):  

        _x = x[i:i + seq_length]
        print(_x.shape)
        _y = y[i + seq_length]  # Next close price
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)

    # train/test split
    train_size = int(len(dataY) * 0.7)  
    test_size = len(dataY) - train_size
    trainX = torch.Tensor(np.array(dataX[0:train_size]))
    trainX = Variable(trainX)
    testX = torch.Tensor(np.array(dataX[train_size:len(dataX)]))
    testX = Variable(testX)
    trainY = torch.Tensor(np.array(dataY[0:train_size]))
    trainY = Variable(trainY)
    testY = torch.Tensor(np.array(dataY[train_size:len(dataY)]))
    testY = Variable(testY)

    # Instantiate RNN model
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)

    # Set loss and optimizer function
    criterion = torch.nn.MSELoss()  # mean-squared error for regression, loss function   
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)  # optimization algorithm  
 

    # Train the model
    for epoch in range(num_epochs):
        outputs = lstm(trainX)
        optimizer.zero_grad()
        # obtain the loss function
        loss = criterion(outputs, trainY)
        loss.backward()  
        optimizer.step()
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    print("Learning finished!")

    # Test the model
    lstm.eval()
    test_predict = lstm(testX)

    # Plot predictions
    test_predict = test_predict.data.numpy()
    testY = testY.data.numpy()
    testY = MinMaxUnscaler(testY, maxValue, minValue)
    test_predict = MinMaxUnscaler(test_predict, maxValue, minValue)
    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.savefig("demo.png")
    plt.plot([1, 2])
    plt.legend(['Real Price', 'Predicted Price'])
    global indicator_g
    global company_name_g
    title = 'Stock Market prediciton for ' + company_name_g + ' with ' + ' '.join(indicator_g) if indicator_g else 'Stock Market prediciton for ' + company_name_g

    plt.title(title)
    plt.show()


def sort(company_name, indicator_name, company_data, technical_data):
    global global_filename

    # Get minimum length of data
    if indicator_name:
        shorter_len = len(company_data) if (len(company_data) < min(map(len, technical_data))) else min(map(len, technical_data))
        end = 1300 if (shorter_len > 1300) else shorter_len
    else:
        end = 1300 if (len(company_data) > 1300) else len(company_data)

    # Create file
    file_name = 'stockMarket - ' + company_name + ' with ' + ' '.join(indicator_name) + '.csv' if indicator_name else 'stockMarket - ' + company_name + '.csv'
    newfile = open(file_name, 'w+')
    print('File ~' + file_name + '~ created')
    global_filename = file_name

    # Write the header of the file
    header = '# Stock Market Information for ' + company_name + '-' + ' '.join(indicator_name) + '\n' + '# Open,High,Low,Volume,' + ','.join(indicator_name) + ',Close' + '\n' if indicator_name else '# Stock Market Information for ' + company_name + '\n' + '# Open,High,Low,Volume,Close\n'
    newfile.write(header)

    # Sort the data and write it in the file
    for i in range(1, end):
        company_array = company_data[i].split(',')  # Divide the company data line in an array

        line = ''
        if len(company_array) == 6:  # Writing company data
            line += company_array[1] + ','  # Writing Open
            line += company_array[2] + ','  # Writing High
            line += company_array[3] + ','  # Writing Low
            line += company_array[5].replace('\r', '')  # Writing volume and delete the \n

        if indicator_name:
            for indic in technical_data:
                technical_array = indic[i].split(',')  # Divide the technical data line in an array
                if len(technical_array) == 2:  # Writing technical indicator data
                    line += ',' + technical_array[1].replace('\r', '')  # Writing close

        if len(company_array) == 6:
            line += ',' + company_array[4]  # Writing Close
        print('len', len(company_array))
        line += '\n'
        newfile.write(line)
    newfile.close()


def getCompanydata(company):
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + company + '&outputsize=full&apikey=' + apikey + '&datatype=csv'
    r = requests.get(url=url)
    content = r.text.split('\n')
    print('size of company data: ', len(content))
    return content


def getTechnicalData(company, type_array):
    global input_size_g
    content = []
    array_len = []
    for type in type_array:
        url = 'https://www.alphavantage.co/query?function=' + type + '&symbol=' + company + '&interval=daily&time_period=200&series_type=close&apikey=' + apikey + '&datatype=csv'
        r = requests.get(url=url)
        content.append(r.text.split('\n'))
        array_len.append(len(r.text.split('\n')))
    input_size_g += len(content)
    print(input_size_g)
    print('size of technical data: ', array_len)
    return content


def main():
    if len(sys.argv) == 2 and sys.argv[1] == '--help':
        print('getCompanyData.py:\n\tDescription: This script will get stock market data for a company and create a file with the data '
              '\n\tUsage: >python3.6 getCompanyData.py [companyName] [technicalIndicator]\n\t\t- companyName has to be the stock market name of the '
              'company\n\t\t- ex: Microsoft = MSFT\n\t\tIf no name is provided, Microsoft Data will be downloaded\n\n\tIf you run the script without '
              'parameter, it will ask you for the symbol of the company and the technical indicator you want.\n\t\tIn case of the company name is '
              'empty, the script will use Microsoft data by default.\n\t\tIn case of the technical indicator is empty, the script will not use any '
              'technical indicator\n\t\tIf you want more than one technical indicator, input your indicators names separated by a space. ex: SMA '
              'MACD\n')
        exit()
    technical_data = False
    indicator_name = False

    if len(sys.argv) == 2:
        company_name = sys.argv[1]
    elif len(sys.argv) == 3:
        company_name = sys.argv[1]
        indicator_name = sys.argv[2]
    else:
        company_name = input("Enter the symbol of your company. ex: MSFT = Microsoft : ")
        indicator_name = input("Enter the name of your technical indicator. ex: SMA =  simple moving average\nIf you don't want "
                               "any technical indicator, let this input empty : ")

    if indicator_name == '':
        indicator_name = False
    else:
        indicator_name = indicator_name.split(' ')

    if company_name == '':
        company_name = 'MSFT'

    global company_name_g
    global indicator_g
    company_name_g = company_name
    indicator_g = indicator_name

    print('\n----- getting information for {} -----\n'.format(company_name))
    company_data = getCompanydata(company_name)
    if indicator_name:
        print('\n----- getting technical information for {} -----\n'.format(' '.join(indicator_name)))
        technical_data = getTechnicalData(company_name, indicator_name)
    print('\n----- sorting information -----\n')
    sort(company_name, indicator_name, company_data, technical_data)
    print('\n----- data created -----\n')
    predictor()


if __name__ == "__main__":
    main()


