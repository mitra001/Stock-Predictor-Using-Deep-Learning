import sys
import requests

apikey = 'IVQTTYEFMDLY30AX'


def sort(company_name, indicator_name, company_data, technical_data):
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

    # Write the header of the file
    header = '# Stock Market Information for ' + company_name + '-' + ' '.join(indicator_name) + '\n' + '# Open,High,Low,Volume,Close,' + ','.join(indicator_name) + '\n' if indicator_name else '# Stock Market Information for ' + company_name + '\n' + '# Open,High,Low,Volume,Close\n'
    newfile.write(header)

    # Sort the data and write it in the file
    for i in range(1, end):
        company_array = company_data[i].split(',')  # Divide the company data line in an array

        line = ''
        if len(company_array) == 6:  # Writing company data
            line += company_array[1] + ','  # Writing Open
            line += company_array[2] + ','  # Writing High
            line += company_array[3] + ','  # Writing Low
            line += company_array[5].replace('\r', '') + ','  # Writing volume and delete the \n
            line += company_array[4]  # Writing Close

        if indicator_name:
            for indic in technical_data:
                # print('indic', indic)
                technical_array = indic[i].split(',')  # Divide the technical data line in an array
                if len(technical_array) == 2:  # Writing technical indicator data
                    line += ',' + technical_array[1].replace('\r', '')  # Writing close

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
    content = []
    array_len = []
    for type in type_array:
        url = 'https://www.alphavantage.co/query?function=' + type + '&symbol=' + company + '&interval=daily&time_period=200&series_type=close&apikey=' + apikey + '&datatype=csv'
        r = requests.get(url=url)
        content.append(r.text.split('\n'))
        array_len.append(len(r.text.split('\n')))
    print('content : #{}#'.format(content))
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
    print(company_name)
    print(indicator_name)
    print('\n----- getting information for {} -----\n'.format(company_name))
    company_data = getCompanydata(company_name)
    if indicator_name:
        print('\n----- getting technical information for {} -----\n'.format(' '.join(indicator_name)))
        technical_data = getTechnicalData(company_name, indicator_name)
    print('\n----- sorting information -----\n')
    sort(company_name, indicator_name, company_data, technical_data)
    print('\n----- finished -----\n')


if __name__ == "__main__":
    main()
