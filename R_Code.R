library(tidyquant)
library(keras)
library(rsample)


############ substring date
date_historic<-Sys.Date()%>%
  as.character()
date_historic<-substr(date_historic,1,4)%>%
  as.numeric()

############ 
############ Input the length of historical data in years
past<-5
############ 
############ 

########### Define historic date
date_historic<-paste0(date_historic-past,substr(as.character(Sys.Date()),5,10))

########### Define Data 
stock<-getSymbols("AAPL",auto.assign = FALSE, from =date_historic,to=Sys.Date())
stock_d<-data.frame(index(stock),as.numeric(stock[,6])) # We take closed adjusted price

lag_1<-Lag(stock_d[,2],k=1)
lag_2<-Lag(stock_d[,2],k=2)
lag_3<-Lag(stock_d[,2],k=3)

stock_d<-data.frame(actual=stock_d[,2],Lag1=lag_1,Lag2=lag_2,Lag3=lag_3)
stock_d<-stock_d[4:nrow(stock_d),]

data_range<-function(x) {(x-min(x))/(max(x)-min(x))}
stock_pp<-as.matrix(sapply(stock_d,data_range))



split<-initial_time_split(as.data.frame(stock_pp),prop=0.6)
nn_train<-training(split)
nn_test<-testing(split)

unscale_data<-function(x,max_x,min_x){x*(max_x-min_x)+min_x}


use_session_with_seed(8)

  xkeras_train<-as.matrix(nn_train[,2:4])
  ykeras_train<-as.matrix(nn_train[,1])
  
  xkeras_test<-as.matrix(nn_test[,2:4])
  ykeras_test<-as.matrix(nn_test[,1])
  
  dim(xkeras_train)<-c(nrow(xkeras_train),ncol(xkeras_train),1)
  dim(xkeras_test)<-c(nrow(xkeras_test),ncol(xkeras_test),1)
  

  
  model<-keras_model_sequential()
  model%>%
    layer_lstm(5,input_shape = c(ncol(xkeras_train),1),activation="relu")%>%
    layer_dense(units=25,activation="relu")%>%
    layer_dense(units=5)%>%
    layer_dense(units=1,activation="linear")
  
  model%>%compile(
    loss="mae",
    optimizer="RMSprop",
    metrics=c("mae") 
  )
  
  model%>%fit(xkeras_train,ykeras_train,epochs=50,batch_size=256,shuffle=F)
  
  #model%>%evaluate(xkeras_train,ykeras_train,verbose=0)
  #model%>%evaluate(xkeras_test,ykeras_test,verbose=0)
  y_pred=model%>%predict(xkeras_test)
  lstm_actual<-unscale_data(y_pred,max(stock_d[,1]),min(stock_d[,1]))
  
plot(stock_d[nrow(nn_train):nrow(stock_d),1],type="l",ylim=c(100,300))
lines(lstm_actual,type="l",col="red")


