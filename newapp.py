from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.uic import loadUi
import sys
import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from  matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT  as  NavigationToolbar)
import pandas as pd
import yfinance as yf

from yahoo_fin import stock_info as si
import pylab
import time
import pyqtgraph
from datetime import date, timedelta
from newsapi import NewsApiClient
from textblob import TextBlob


global input_stock, input_stock2, input_stock3, forecast_out1, df
global e_date, s_date, sdate, edate, e_date2, s_date2, sdate2, edate2, e_date3, s_date3, sdate3, edate3
class MainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)

        loadUi("GUI4.ui", self)

        self.setWindowTitle("MoneyGrow")

        self.pushButton2.clicked.connect(self.update_graph)
        self.pushButton1.clicked.connect(self.update_table)
        self.pushButton4.clicked.connect(self.update_graph1)
        self.pushButton5.clicked.connect(self.update_graph2)
        self.radioButtonclose.toggled.connect(self.update_graph3)
        self.radioButtonhigh.toggled.connect(self.update_graph4)
        self.radioButtonopen.toggled.connect(self.update_graph5)
        self.radioButtonv.toggled.connect(self.update_graph6)
        self.radioButtonlow.toggled.connect(self.update_graph7)
        self.radioButtonrm.toggled.connect(self.update_graph8)
        self.radioButtonema.toggled.connect(self.update_graph9)
        self.radioButtonma.toggled.connect(self.update_graph21)
        self.radioButtonmm.toggled.connect(self.update_graph22)
        self.radioButtonroc.toggled.connect(self.update_graph23)
        self.radioButtonb.toggled.connect(self.update_graph24)

        self.pushButtoncd.clicked.connect(self.update1)
        self.btnadd2.clicked.connect(self.update2)


        self.btnadd.clicked.connect(self.update)
        self.test.plotItem.showGrid(True, True, 0.7)
        self.test.setTitle("LIVE STOCK PRICE")
        self.test.setLabel('left', 'STOCK PRICE', color='white', size=30)
        self.test.showGrid(x=True, y=True)
        
        self.pushButtonnews.clicked.connect(self.updatenews)






        #self.radioButtonsh.toggled.connect(self.update_graph10)
        #self.radioButtonf.toggled.connect(self.update_graph11)
        #self.radioButtonbs.toggled.connect(self.update_graph12)
        #self.radioButtoncf.toggled.connect(self.update_graph13)
        #self.radioButtone.toggled.connect(self.update_graph14)
        #self.radioButtonr.toggled.connect(self.update_graph15)
        #self.radioButtongi.toggled.connect(self.update_graph16)
        #self.radioButtona.toggled.connect(self.update_graph17)
        #self.radioButtonaise.toggled.connect(self.update_graph18)

























        self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))

    def update_graph(self):
        input_stock = self.lineEdit1.text()
        #input_stock="AMZN"
        s_date = self.dateEdit1.date()
        sdate = s_date.toPyDate()
        e_date = self.dateEdit2.date()
        edate = e_date.toPyDate()

        to = yf.Ticker(input_stock)
        df = to.history(start=sdate, end=edate)
        df1 = df[['Close']]
        df1.reset_index(level=0, inplace=True)
        df1.columns = ['ds', 'y']


        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.plot(df1.ds, df1.y,'m')
        #self.MplWidget.canvas.axes.plot(t, a)
        self.MplWidget.canvas.axes.legend('Close', loc='upper left')
        self.MplWidget.canvas.axes.set_title(' Company Stock Trend')
        self.MplWidget.canvas.draw()

    def update_table(self):
        input_stock = self.lineEdit1.text()
        # input_stock="AMZN"
        s_date = self.dateEdit1.date()
        sdate = s_date.toPyDate()
        e_date = self.dateEdit2.date()
        edate = e_date.toPyDate()

        to = yf.Ticker(input_stock)
        df = to.history(start=sdate, end=edate)
        df1=df.reset_index()

        self.model = pandasModel(df1)
        self.tableView1.setModel(self.model)
        self.tableView1.show()

    def update_graph1(self):
        input_stock2 = self.lineEdit2.text()
        s_date2 = self.dateEdit3.date()
        sdate2 = s_date2.toPyDate()
        e_date2 = self.dateEdit4.date()
        edate2 = e_date2.toPyDate()

        to = yf.Ticker(input_stock2)
        df = to.history(start=sdate2, end=edate2)


        dfx=df
        df = df[['Close']]
        forecast_out = self.spinBox1.value()
        df['Prediction'] = df[['Close']].shift(-forecast_out)
        X = np.array(df.drop(['Prediction'], 1))
        # Remove the last 'n' rows
        X = X[:-forecast_out]
        y = np.array(df['Prediction'])
        # Get all of the y values except the last 'n' rows
        y = y[:-forecast_out]

        rng = pd.date_range(edate2, periods=forecast_out, freq='B')
        df2 = pd.DataFrame({'Date': rng, 'Val': np.random.randn(len(rng))})
        df3 = df2.reset_index()
        df4 = df3[['Date']]
        t = np.asarray(df4)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        svr_rbf.fit(x_train, y_train)
        svm_confidence = svr_rbf.score(x_test, y_test)
        svm_ac=svm_confidence*100
        svm_acc=str(svm_ac)
        self.labelacc.setStyleSheet('color:red')
        self.labelacc.setText("THE ACCURACY OF THE PREDICTION IS: "+ svm_acc)
        x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
        svm_prediction = svr_rbf.predict(x_forecast)
       # plt.plot(x_forecast, svm_prediction)

        self.MplWidget2.canvas.axes.clear()
        self.MplWidget2.canvas.axes.plot(t, svm_prediction,'r-o')
        #self.MplWidget.canvas.axes.plot(t, a)
        self.MplWidget2.canvas.axes.legend('Adj. Close', loc='upper right')
        self.MplWidget2.canvas.axes.set_title(' Company Stock Trend')
        self.MplWidget2.canvas.draw()

    def update_graph2(self):
        input_stock2 = self.lineEdit2.text()
        s_date2 = self.dateEdit3.date()
        sdate2 = s_date2.toPyDate()
        e_date2 = self.dateEdit4.date()
        edate2 = e_date2.toPyDate()

        to = yf.Ticker(input_stock2)
        df = to.history(start=sdate2, end=edate2)

        dfx=df
        df = df[['Close']]
        forecast_out = self.spinBox1.value()
        df['Prediction'] = df[['Close']].shift(-forecast_out)
        X = np.array(df.drop(['Prediction'], 1))
        # Remove the last 'n' rows
        X = X[:-forecast_out]
        y = np.array(df['Prediction'])
        # Get all of the y values except the last 'n' rows
        y = y[:-forecast_out]

        rng = pd.date_range(edate2, periods=forecast_out, freq='B')
        df2 = pd.DataFrame({'Date': rng, 'Val': np.random.randn(len(rng))})
        df3 = df2.reset_index()
        df4 = df3[['Date']]
        t = np.asarray(df4)

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        lr_confidence = lr.score(x_test, y_test)
        lr_ac = lr_confidence * 100
        lr_acc = str(lr_ac)
        self.labelacc.setStyleSheet('color:red')
        self.labelacc.setText("THE ACCURACY OF THE PREDICTION IS: " + lr_acc)
        x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
        lr_prediction = lr.predict(x_forecast)
       # plt.plot(x_forecast, svm_prediction)

        self.MplWidget2.canvas.axes.clear()
        self.MplWidget2.canvas.axes.plot(t, lr_prediction,'g-o')
        #self.MplWidget.canvas.axes.plot(t, a)
        self.MplWidget2.canvas.axes.legend('Adj. Close', loc='upper right')
        self.MplWidget2.canvas.axes.set_title(' Company Stock Trend')
        self.MplWidget2.canvas.draw()

    def update_graph3(self):
        input_stock3 = self.lineEdit4.text()
        s_date3 = self.dateEdit5.date()
        sdate3 = s_date3.toPyDate()
        e_date3 = self.dateEdit6.date()
        edate3 = e_date3.toPyDate()

        to = yf.Ticker(input_stock3)
        df = to.history(start=sdate3, end=edate3)

        df1 = df
        df1 = df[['Close']]
        df1.reset_index(level=0, inplace=True)
        df1.columns = ['ds', 'y']

        self.MplWidget3.canvas.axes.clear()
        self.MplWidget3.canvas.axes.plot(df1.ds, df1.y, 'm')
        # self.MplWidget.canvas.axes.plot(t, a)
        self.MplWidget3.canvas.axes.legend('Close', loc='upper right')
        self.MplWidget3.canvas.axes.set_title(' Company Stock Trend')
        self.MplWidget3.canvas.draw()

    def update_graph4(self):
        input_stock3 = self.lineEdit4.text()
        s_date3 = self.dateEdit5.date()
        sdate3 = s_date3.toPyDate()
        e_date3 = self.dateEdit6.date()
        edate3 = e_date3.toPyDate()

        to = yf.Ticker(input_stock3)
        df = to.history(start=sdate3, end=edate3)

        df1 = df
        df1 = df[['High']]
        df1.reset_index(level=0, inplace=True)
        df1.columns = ['ds', 'y']


        self.MplWidget3.canvas.axes.clear()
        self.MplWidget3.canvas.axes.plot(df1.ds, df1.y, 'orange')
        # self.MplWidget.canvas.axes.plot(t, a)
        self.MplWidget3.canvas.axes.legend('High', loc='upper right')
        self.MplWidget3.canvas.axes.set_title(' Company Stock Trend')
        self.MplWidget3.canvas.draw()

    def update_graph5(self):
        input_stock3 = self.lineEdit4.text()
        s_date3 = self.dateEdit5.date()
        sdate3 = s_date3.toPyDate()
        e_date3 = self.dateEdit6.date()
        edate3 = e_date3.toPyDate()

        to = yf.Ticker(input_stock3)
        df = to.history(start=sdate3, end=edate3)

        df1 = df
        df1 = df[['Open']]
        df1.reset_index(level=0, inplace=True)
        df1.columns = ['ds', 'y']

        self.MplWidget3.canvas.axes.clear()
        self.MplWidget3.canvas.axes.plot(df1.ds, df1.y, 'c')
        # self.MplWidget.canvas.axes.plot(t, a)
        self.MplWidget3.canvas.axes.legend('Open', loc='upper right')
        self.MplWidget3.canvas.axes.set_title(' Company Stock Trend')
        self.MplWidget3.canvas.draw()

    def update_graph6(self):
        input_stock3 = self.lineEdit4.text()
        s_date3 = self.dateEdit5.date()
        sdate3 = s_date3.toPyDate()
        e_date3 = self.dateEdit6.date()
        edate3 = e_date3.toPyDate()

        to = yf.Ticker(input_stock3)
        df = to.history(start=sdate3, end=edate3)

        df1 = df
        df1 = df[['Volume']]
        df1.reset_index(level=0, inplace=True)
        df1.columns = ['ds', 'y']

        self.MplWidget3.canvas.axes.clear()
        self.MplWidget3.canvas.axes.plot(df1.ds, df1.y, 'brown')
        # self.MplWidget.canvas.axes.plot(t, a)
        self.MplWidget3.canvas.axes.legend('Volume', loc='upper right')
        self.MplWidget3.canvas.axes.set_title(' Company Stock Trend')
        self.MplWidget3.canvas.draw()

    def update_graph7(self):
        input_stock3 = self.lineEdit4.text()
        s_date3 = self.dateEdit5.date()
        sdate3 = s_date3.toPyDate()
        e_date3 = self.dateEdit6.date()
        edate3 = e_date3.toPyDate()

        to = yf.Ticker(input_stock3)
        df = to.history(start=sdate3, end=edate3)

        df1 = df
        df1 = df[['Low']]
        df1.reset_index(level=0, inplace=True)
        df1.columns = ['ds', 'y']

        self.MplWidget3.canvas.axes.clear()
        self.MplWidget3.canvas.axes.plot(df1.ds, df1.y, 'pink')
        # self.MplWidget.canvas.axes.plot(t, a)
        self.MplWidget3.canvas.axes.legend('Low', loc='upper left')
        self.MplWidget3.canvas.axes.set_title(' Company Stock Trend')
        self.MplWidget3.canvas.draw()

    def update_graph8(self):
        input_stock3 = self.lineEdit4.text()
        s_date3 = self.dateEdit5.date()
        sdate3 = s_date3.toPyDate()
        e_date3 = self.dateEdit6.date()
        edate3 = e_date3.toPyDate()

        to = yf.Ticker(input_stock3)
        df = to.history(start=sdate3, end=edate3)

        df1 = df
        df1 = df[['Close']]
        df1.reset_index(level=0, inplace=True)
        df1.columns = ['ds', 'y']

        str1="The SMA is a technical indicator for determining if an asset price will continue or reverse a bull or bear trend. The SMA is calculated as the arithmetic average of an asset's price over some period. The SMA can be enhanced as an exponential moving average (EMA) that more heavily weights recent price action."
        self.labelwiki.setText(str1)

        rolling_mean = df1.y.rolling(window=20).mean()
        rolling_mean1 = df1.y.rolling(window=50).mean()

        self.MplWidget3.canvas.axes.clear()
        #self.MplWidget3.canvas.axes.plot(df1.ds, df1.y, 'm')
        self.MplWidget3.canvas.axes.plot(df1.ds, rolling_mean,label='20 day SMA',color='orange')
        self.MplWidget3.canvas.axes.plot(df1.ds, rolling_mean1, label='50 day SMA', color='magenta')
        self.MplWidget3.canvas.axes.legend(loc='upper left')
        self.MplWidget3.canvas.axes.set_title(' Company Stock Trend')
        self.MplWidget3.canvas.draw()

    def update_graph9(self):
        input_stock3 = self.lineEdit4.text()
        s_date3 = self.dateEdit5.date()
        sdate3 = s_date3.toPyDate()
        e_date3 = self.dateEdit6.date()
        edate3 = e_date3.toPyDate()

        to = yf.Ticker(input_stock3)
        df = to.history(start=sdate3, end=edate3)

        df1 = df
        df1 = df[['Close']]
        df1.reset_index(level=0, inplace=True)
        df1.columns = ['ds', 'y']

        str1="The EMA is a moving average that places a greater weight and significance on the most recent data points. Like all moving averages, this technical indicator is used to produce buy and sell signals based on crossovers and divergences from the historical average."
        self.labelwiki.setText(str1)

        exp1 = df1.y.ewm(span=20, adjust=False).mean()
        exp2 = df1.y.ewm(span=50, adjust=False).mean()

        self.MplWidget3.canvas.axes.clear()
       # self.MplWidget3.canvas.axes.plot(df1.ds, df1.y, 'm')
        self.MplWidget3.canvas.axes.plot(df1.ds, exp1,label='20 day EMA',color='r')
        self.MplWidget3.canvas.axes.plot(df1.ds, exp2, label='50 day EMA', color='purple')
        self.MplWidget3.canvas.axes.legend(loc='upper left')
        self.MplWidget3.canvas.axes.set_title(' Company Stock Trend')
        self.MplWidget3.canvas.draw()

    def update_graph10(self):
        input_stock4 = self.lineEditnew.text()
        to = yf.Ticker(input_stock4)
        df=to.major_holders
        df.reset_index(level=0, inplace=True)
        self.model = pandasModel(df)
        self.tableViewnew.setModel(self.model)
        self.tableViewnew.show()

    def update_graph11(self):
        input_stock4 = self.lineEditnew.text()
        to = yf.Ticker(input_stock4)
        st=to.financials
        st.reset_index(level=0, inplace=True)
        self.model = pandasModel(st)
        self.tableViewnew.setModel(self.model)
        self.tableViewnew.show()

    def update_graph12(self):
        input_stock4 = self.lineEditnew.text()
        to = yf.Ticker(input_stock4)
        st=to.balance_sheet
        st.reset_index(level=0, inplace=True)
        self.model = pandasModel(st)
        self.tableViewnew.setModel(self.model)
        self.tableViewnew.show()

    def update_graph13(self):
        input_stock4 = self.lineEditnew.text()
        to = yf.Ticker(input_stock4)
        st=to.cashflow
        st.reset_index(level=0, inplace=True)
        self.model = pandasModel(st)
        self.tableViewnew.setModel(self.model)
        self.tableViewnew.show()

    def update_graph14(self):
        input_stock4 = self.lineEditnew.text()
        to = yf.Ticker(input_stock4)
        st=to.earnings
        st.reset_index(level=0, inplace=True)
        self.model = pandasModel(st)
        self.tableViewnew.setModel(self.model)
        self.tableViewnew.show()

    def update_graph15(self):
        input_stock4 = self.lineEditnew.text()
        to = yf.Ticker(input_stock4)
        st=to.recommendations
       # st.reset_index(level=0, inplace=True)
        self.model = pandasModel(st)
        self.tableViewnew.setModel(self.model)
        self.tableViewnew.show()

    def update_graph16(self):
        input_stock4 = self.lineEditnew.text()
        to = yf.Ticker(input_stock4)
        st=to.dividends
     #   st.reset_index(level=0, inplace=True)
        self.model = pandasModel(st)
        self.tableViewnew.setModel(self.model)
        self.tableViewnew.show()

    def update_graph17(self):
        input_stock4 = self.lineEditnew.text()
        to = yf.Ticker(input_stock4)
        st=to.actions
        self.model = pandasModel(st)
        self.tableViewnew.setModel(self.model)
        self.tableViewnew.show()

    def update_graph18(self):
        input_stock4 = self.lineEditnew.text()
        to = yf.Ticker(input_stock4)
        st=to.calendar
        st.reset_index(level=0, inplace=True)
        self.model = pandasModel(st)
        self.tableViewnew.setModel(self.model)
        self.tableViewnew.show()

    def update_graph21(self):
        input_stock3 = self.lineEdit4.text()
        s_date3 = self.dateEdit5.date()
        sdate3 = s_date3.toPyDate()
        e_date3 = self.dateEdit6.date()
        edate3 = e_date3.toPyDate()

        to = yf.Ticker(input_stock3)
        df = to.history(start=sdate3, end=edate3)
        n = self.spinBox2.value()

        str1="A moving average (MA) is a widely used indicator in technical analysis that helps smooth out price action by filtering out the “noise” from random short-term price fluctuations. It is a trend-following, or lagging, indicator because it is based on past prices.The most common applications of moving averages are to identify the trend direction and to determine support and resistance levels."
        self.labelwiki.setText(str1)
        dfx = moving_average(df, n)
        dfy = dfx['MA_'+ str(n)].dropna()
        dfy.to_frame()
        dfw = dfy.reset_index()
        dfw.reset_index()
        dfw.columns = ['ds', 'y']

        self.MplWidget3.canvas.axes.clear()
        self.MplWidget3.canvas.axes.plot(dfw.ds, dfw.y, label= str(n) + ' day Moving Average', color='pink')
       # self.MplWidget3.canvas.axes.plot(dfl.ds, dfl.y, label='50 day Moving Average', color='darkgoldenrod')
        self.MplWidget3.canvas.axes.legend(loc='upper left')
        self.MplWidget3.canvas.axes.set_title(' Company Stock Trend')
        self.MplWidget3.canvas.draw()

    def update_graph22(self):
        input_stock3 = self.lineEdit4.text()
        s_date3 = self.dateEdit5.date()
        sdate3 = s_date3.toPyDate()
        e_date3 = self.dateEdit6.date()
        edate3 = e_date3.toPyDate()

        to = yf.Ticker(input_stock3)
        df = to.history(start=sdate3, end=edate3)
        n = self.spinBox2.value()

        str1="The Momentum indicator compares where the current price is in relation to where the price was in the past. How far in the past the comparison is made is up to the technical analysis trader. The calculation of Momentum is quite simple (n is the number of periods the technical trader selects): [Current Price]-n"
        self.labelwiki.setText(str1)
        dfx = momentum(df, n)
        dfy = dfx['Momentum_'+ str(n)].dropna()
        dfy.to_frame()
        dfw = dfy.reset_index()
        dfw.reset_index()
        dfw.columns = ['ds', 'y']

        self.MplWidget3.canvas.axes.clear()
        self.MplWidget3.canvas.axes.plot(dfw.ds, dfw.y, label= str(n) + ' day Momentum', color='darkgoldenrod')
       # self.MplWidget3.canvas.axes.plot(dfl.ds, dfl.y, label='50 day Moving Average', color='darkgoldenrod')
        self.MplWidget3.canvas.axes.legend(loc='upper left')
        self.MplWidget3.canvas.axes.set_title(' Company Stock Trend')
        self.MplWidget3.canvas.draw()

    def update_graph23(self):
        input_stock3 = self.lineEdit4.text()
        s_date3 = self.dateEdit5.date()
        sdate3 = s_date3.toPyDate()
        e_date3 = self.dateEdit6.date()
        edate3 = e_date3.toPyDate()

        to = yf.Ticker(input_stock3)
        df = to.history(start=sdate3, end=edate3)
        n = self.spinBox2.value()

        str1="The Price Rate of Change (ROC) is a momentum-based technical indicator that measures the percentage change in price between the current price and the price a certain number of periods ago. The ROC indicator is plotted against zero, with the indicator moving upwards into positive territory if price changes are to the upside, and moving into negative territory if price changes are to the downside."
        self.labelwiki.setText(str1)
        dfx = rate_of_change(df, n)
        dfy = dfx['ROC_'+ str(n)].dropna()
        dfy.to_frame()
        dfw = dfy.reset_index()
        dfw.reset_index()
        dfw.columns = ['ds', 'y']

        self.MplWidget3.canvas.axes.clear()
        self.MplWidget3.canvas.axes.plot(dfw.ds, dfw.y, label= str(n) + ' day Rate of Change', color='darkgoldenrod')
       # self.MplWidget3.canvas.axes.plot(dfl.ds, dfl.y, label='50 day Moving Average', color='darkgoldenrod')
        self.MplWidget3.canvas.axes.legend(loc='upper left')
        self.MplWidget3.canvas.axes.set_title(' Company Stock Trend')
        self.MplWidget3.canvas.draw()

    def update_graph24(self):
        input_stock3 = self.lineEdit4.text()
        s_date3 = self.dateEdit5.date()
        sdate3 = s_date3.toPyDate()
        e_date3 = self.dateEdit6.date()
        edate3 = e_date3.toPyDate()

        to = yf.Ticker(input_stock3)
        df = to.history(start=sdate3, end=edate3)
        n = self.spinBox2.value()

        str1="A Bollinger Band® is a technical analysis tool defined by a set of lines plotted two standard deviations (positively and negatively) away from a simple moving average (SMA) of the security's price, but can be adjusted to user preferences."
        self.labelwiki.setText(str1)

        dfx = bollinger_bands(df, n)
        dfy = dfx['BollingerB_' + str(n)].dropna()
        dfy.to_frame()
        dfw = dfy.reset_index()
        dfw.reset_index()
        dfw.columns = ['ds', 'y']

        self.MplWidget3.canvas.axes.clear()
        self.MplWidget3.canvas.axes.plot(dfw.ds, dfw.y, label=str(n) + ' day Bollinger Bands', color='magenta')
        # self.MplWidget3.canvas.axes.plot(dfl.ds, dfl.y, label='50 day Moving Average', color='darkgoldenrod')
        self.MplWidget3.canvas.axes.legend(loc='upper left')
        self.MplWidget3.canvas.axes.set_title(' Company Stock Trend')
        self.MplWidget3.canvas.draw()

    def update(self):
        input_live = self.lineEditlive.text()
        a = [int(si.get_live_price(input_live))]

        points = 100  # number of data points
        a.append(int(si.get_live_price(input_live)))
        X = a
        # Y = np.sin(np.arange(points) / points * 3 * np.pi + time.time())
        C = pyqtgraph.hsvColor(time.time() / 5 % 1, alpha=.5)
        pen = pyqtgraph.mkPen(color=C, width=5)
        self.test.plot(X, pen=pen, clear=True)
        # print("update took %.02f ms" % ((time.clock() - t1) * 1000))
        if self.test1.isChecked():
            QtCore.QTimer.singleShot(1, self.update)  # QUICKLY repeat

    def update1(self):
        input_stocklv = self.lineEditlive.text()


        df=si.get_quote_table(input_stocklv, dict_result=False)
        a = df['value'][15] - df['value'][14]
        #print(round(a, 2))
        b = (a / df['value'][15]) * 100
        #print(round(b, 2), "%")
        c= "(" + str(round(b,2)) + "%" + ")"

        if df['value'][14] > df['value'][15]:
            self.qplabel.setStyleSheet('color:red')
            self.ndlabel.setStyleSheet('color:red')
            self.nplabel.setStyleSheet('color:red')
            self.qplabel.setText(str(df['value'][15]))
            self.ndlabel.setText(str(round(a,2)))
            self.nplabel.setText(c)
        elif df['value'][14] < df['value'][15]:
            self.qplabel.setStyleSheet('color:green')
            self.ndlabel.setStyleSheet('color:green')
            self.nplabel.setStyleSheet('color:green')
            self.qplabel.setText(str(df['value'][15]))
            self.ndlabel.setText(str(round(a,2)))
            self.nplabel.setText(c)



        self.model = pandasModel(df)
        self.tableViewcd.setModel(self.model)
        self.tableViewcd.show()

    def update2(self):
        input_live = self.lineEditlive.text()
        to = yf.Ticker(input_live)
        today = date.today()
        yesterday = today - timedelta(days=7)
        df = to.history(start=yesterday, end=today)

        df1 = df[['Close']]
        df1.reset_index(level=0, inplace=True)
        df1.columns = ['ds', 'y']

        self.MplWidgetlive.canvas.axes.clear()
        self.MplWidgetlive.canvas.axes.plot(df1.ds, df1.y, 'm', label='Close')
       # self.MplWidgetlive.canvas.axes.plot(t, a)
        self.MplWidgetlive.canvas.axes.legend(loc='upper left')
        self.MplWidgetlive.canvas.axes.set_title(' Past Week Stock Price')
        self.MplWidgetlive.canvas.draw()

    def updatenews(self):
        input_news = self.lineEditnews.text()
        n=self.spinBoxnews.value()

        today = date.today()
        yesterday = today - timedelta(days=7)

        newsapi = NewsApiClient(api_key='ce8e0ad3dc304cb685d2f12ea288c8b6')

        data = newsapi.get_everything(q=input_news,
                                      sources='bbc-news,the-verge,daily-mail,financial-post,abc-news, financial-post,cnbc',
                                      domains='bbc.co.uk,economist.com,ft.com,moneycontrol.com',
                                      from_param=yesterday,
                                      to=today,
                                      language='en',
                                      sort_by='relevancy',
                                      page_size=n,
                                      page=1)
        articles = data['articles']
        dnews = pd.DataFrame(articles)
        dnews1 = dnews.drop(['author', 'url', 'urlToImage'], axis=1)
        dnews1['Polarity'] = dnews['title'].map(lambda x: sentiment_analysis(x))
        dnews1['Subjectivity'] = dnews['title'].map(lambda x: sentiment_subjectivity(x))


        self.model = pandasModel(dnews1)
        self.tableViewnews.setModel(self.model)
        self.tableViewnews.show()

    def updatenews(self):
        input_news = self.lineEditnews.text()
        n=self.spinBoxnews.value()
        m = self.spinBoxnews1.value()

        today = date.today()
        yesterday = today - timedelta(days=m)

        newsapi = NewsApiClient(api_key='ce8e0ad3dc304cb685d2f12ea288c8b6')

        data = newsapi.get_everything(q=input_news,
                                      sources='bbc-news,the-verge,daily-mail,financial-post,abc-news, financial-post,cnbc',
                                      domains='bbc.co.uk,economist.com,ft.com,moneycontrol.com',
                                      from_param=yesterday,
                                      to=today,
                                      language='en',
                                      sort_by='relevancy',
                                      page_size=n,
                                      page=1)
        articles = data['articles']
        dnews = pd.DataFrame(articles)
        dnews1=dnews.drop(['author', 'url', 'urlToImage', 'content'], axis=1)
        dnews1['Polarity'] = dnews['title'].map(lambda x: sentiment_analysis(x))
        dnews1['Subjectivity'] = dnews['title'].map(lambda x: sentiment_subjectivity(x))


        self.model = pandasModel(dnews1)
        self.tableViewnews.setModel(self.model)
        self.tableViewnews.show()





class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


def moving_average(df, n):
    MA = pd.Series(df['Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
    df = df.join(MA)
    return df

def momentum(df, n):
    M = pd.Series(df['Close'].diff(n), name='Momentum_' + str(n))
    df = df.join(M)
    return df


def rate_of_change(df, n):
    M = df['Close'].diff(n - 1)
    N = df['Close'].shift(n - 1)
    ROC = pd.Series(M / N, name='ROC_' + str(n))
    df = df.join(ROC)
    return df


def bollinger_bands(df, n):
    MA = pd.Series(df['Close'].rolling(n, min_periods=n).mean())
    MSD = pd.Series(df['Close'].rolling(n, min_periods=n).std())
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name='BollingerB_' + str(n))
    df = df.join(B1)
    b2 = (df['Close'] - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name='Bollinger%b_' + str(n))
    df = df.join(B2)
    return df

def sentiment_subjectivity(t):
    text=TextBlob(t)
    texts=text.sentiment
    return texts.subjectivity

def sentiment_analysis(t):
    text=TextBlob(t)
    texts=text.sentiment
    return texts.polarity



app = QApplication([])
window = MainWindow()
window.show()
app.exec_()