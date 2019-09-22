library(tidyquant)
library(keras)


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
stock<-data.frame(index(stock),stock[,6]) # We take closed adjusted price



data_range<-function(x) {(x-min(x))/(max(x)-min(x))}
stock_pp<-as.matrix(sapply(lag_dr_df,data_range))

