# Predicting-Stock-Price-with-LSTM
Predicting stock price with the help of LSTM of the keras package in R


First we need to load the library. These include the package "tidyquant", "keras" and "rsample". 
```{r}
library(tidyquant)
library(keras)
library(rsample)
```

To predict a value, one needs historical data in order to build a model. The more historical data available, the better is the prediction. In this case the historical data contains 5 years of data from current date.
```{r}
date_historic<-Sys.Date()%>%
  as.character()
date_historic<-substr(date_historic,1,4)%>%
  as.numeric()
past<-5
```
We are going to use the stock price of Apple stocks. The required data is downloaded and its size is based on the "past" variable value.
```{r}
date_historic<-paste0(date_historic-past,substr(as.character(Sys.Date()),5,10))

stock<-getSymbols("AAPL",auto.assign = FALSE, from =date_historic,to=Sys.Date())
stock_d<-data.frame(index(stock),as.numeric(stock[,6])) 
```
To model the stock price, we are using 3 lags of the data. That means the model will predict the next day stock price based on the data from present date and 2 days before.
```{r}
lag_1<-Lag(stock_d[,2],k=1)
lag_2<-Lag(stock_d[,2],k=2)
lag_3<-Lag(stock_d[,2],k=3)
stock_d<-data.frame(actual=stock_d[,2],Lag1=lag_1,Lag2=lag_2,Lag3=lag_3)
stock_d<-stock_d[4:nrow(stock_d),]
```
In this step we prepare the LSTM model. To predict with a neural network algorithm, one needs to preprocess the data into a small range. In this case we use the Min-Max method.
```{r}
data_range<-function(x) {(x-min(x))/(max(x)-min(x))}
stock_pp<-as.matrix(sapply(stock_d,data_range))
unscale_data<-function(x,max_x,min_x){x*(max_x-min_x)+min_x}
```
The final step before running the algorithim is to split the data set into a training and testing set. For this example the training set consists of 60% of the data.
```{r}
split<-initial_time_split(as.data.frame(stock_pp),prop=0.6)
nn_train<-training(split)
nn_test<-testing(split)
```
To use the LSTM algorithm, the data must be in three-dimensional array, in which the first includes the number of the observations, the second includes the number of lags and the third defines wether the time series is univariate or not.
```{r}
use_session_with_seed(8)

  xkeras_train<-as.matrix(nn_train[,2:4])
  ykeras_train<-as.matrix(nn_train[,1])
  
  xkeras_test<-as.matrix(nn_test[,2:4])
  ykeras_test<-as.matrix(nn_test[,1])
  
  dim(xkeras_train)<-c(nrow(xkeras_train),ncol(xkeras_train),1)
  dim(xkeras_test)<-c(nrow(xkeras_test),ncol(xkeras_test),1)
```
The LSTM algorithm uses 2 lays of 50 and 25 nodes respectivly. The activation function is the rectified linear unit function. The loss function is the mean absolute error and the gradient descent algorithm the resilient propagation.
```{r}  
  model<-keras_model_sequential()
  model%>%
    layer_lstm(50,input_shape = c(ncol(xkeras_train),1),activation="relu")%>%
    layer_dense(units=25,activation="relu")%>%
    layer_dense(units=1,activation="linear")
  
  model%>%compile(
    loss="mae",
    optimizer="RMSprop",
    metrics=c("mae") 
  )
```
The model is trained in 50 epochs with batch size of 256. The predicted values are unscaled to their original form. 
```{r} 
  model%>%fit(xkeras_train,ykeras_train,epochs=50,batch_size=256,shuffle=F)
  y_pred=model%>%predict(xkeras_test)
  lstm_actual<-unscale_data(y_pred,max(stock_d[,1]),min(stock_d[,1]))
```
To see the results one can do a simple line plot in which the actual and predicted stock price are compared.
```{r} 
plot(stock_d[nrow(nn_train):nrow(stock_d),1],type="l",ylim=c(100,300),xlab="time",ylab="stock price")
lines(lstm_actual,type="l",col="red")
```
In This case the predicted stock price is in red color while the actual in black.
https://github.com/tejinder1988/Predicting-Stock-Price-with-LSTM/blob/master/Rplot.png
