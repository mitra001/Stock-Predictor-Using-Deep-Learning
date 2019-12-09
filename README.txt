This script will get the stock market data for a company and will use part of that data to train himself and part of that data to predict.

Usage: python3.6 predictor_final.py [companyName] [technicalIndicator]

companyName has to be the stock market nsymbol of the company ex: Microsoft = MSFT. 
If no name is provided, Microsoft Data will be downloaded
technicallIndicator has to be any combination of those four: SMA EMA OBV RSI


If you run the script without parameter, it will ask you for the symbol of the company and the technical indicators. 
In case of the company name is empty, the script will use Microsoft data by default.
In case of the technical indicator is empty, the script will not use any technical indicator

If you want more than one technical indicator, input your indicators names separated by a space. ex: SMA RSI