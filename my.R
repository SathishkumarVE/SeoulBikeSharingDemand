library(rpart)              # CART algorithm for decision trees
library(partykit)           # Plotting trees
library(gbm)                  # Boosting algorithms
library(doParallel)     # parallel processing
library(pROC)                 # plot the ROC curve
library(corrplot)  
library(psych)
library(Hmisc)
library(psych)
library(lubridate)
library(RRF)
require(gridExtra)
library(caret)
library(dummies)
library(Metrics)
library(gridBase)
library(gridExtra)
library(lattice)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
if (Sys.getenv("JAVA_HOME")!="")
  Sys.setenv(JAVA_HOME="")
library(rJava)
library(extraTrees)
library(Cubist)
library(gam)
library(e1071)
library(kernlab)

MAPE <- function(y, yhat) {
  mean(abs((y - yhat)/y))
}

MAE <- function(y, yhat) {
  mean(abs((y - yhat)))
}
MSE <- function(y,yhat)
{
  mean((y-yhat)**2)
}
MAPE2 <- function(y, yhat) {
  
  n <- length(y)
  su <- 0
  for (i in 1:n) {
    su <-su+ abs((y[i]-yhat[i])/y[i])
    
  }
  print (su/n)
}



MAE2 <- function(y, yhat) {
  
  n <- length(y)
  su <- 0
  for (i in 1:n) {
    su <-su+ abs((y[i]-yhat[i]))
    
  }
  print (su/n)
}

Rsqu <- function(y, yhat) {
  
  mdata <- mean(y)
  
  1 - sum((y-yhat)^2)/(sum((y-mdata)^2))
  
}

Rmse_f <- function(y, yhat) {
  
  sqrt(mean((y-yhat)^2))
  
}

mbe <- function(y, yhat) {
  mdata <- mean(y)
  mean((y-yhat))/mdata
  
}

cv <- function(y, yhat) {
  mdata <- mean(y)
  sqrt(mean((y-yhat)^2))/mdata
  
}


fitControl <- trainControl(method = "repeatedcv", # cv
                           number=10,repeats=3,  # 10  
                           verboseIter = TRUE,returnResamp = "all")


bike_data <- read.csv("SeoulBikeDataFinal.csv")
bike_data$Date <- strptime(as.character(bike_data$Date),format="%Y-%m-%d")
bike_data$Date <- as.POSIXct(bike_data$Date,tz = "UTC")
class(bike_data$Date)
names(bike_data)
names(bike_data)[2] <- "Count"
names(bike_data)[4] <- "Temp"
names(bike_data)[5] <- "Hum"
names(bike_data)[6] <- "Wind"
names(bike_data)[7] <- "Visb"
names(bike_data)[8] <- "Dew"
names(bike_data)[9] <- "solar"
names(bike_data)[10] <- "Rain"
names(bike_data)[11] <- "Snow"
names(bike_data)[14]<-"Fday"
names(bike_data)
str(bike_data)

weekend_weekday <- function(x) {
  val <- weekdays(x)
  if (val == "Saturday" | val == "Sunday") {
    val2 = "Weekend"
  }
  else {
    val2= "Weekday"
  }
  return(val2)
}

bike_data$WeekStatus <- unlist(lapply(bike_data$Date,weekend_weekday))
bike_data$Day_of_week <-weekdays(bike_data$Date)
unique(bike_data$WeekStatus)
unique(bike_data$Day_of_week)
class(bike_data$Day_of_week)
bike_data$Day_of_week <-as.factor(bike_data$Day_of_week)
bike_data$WeekStatus <- as.factor(bike_data$WeekStatus)
str(bike_data)
dim(bike_data)
names(bike_data)
names(bike_data)[15] <- "Wstatus"
names(bike_data)[16] <- "Dweek"
plot1 <-qplot(bike_data$Date,bike_data$Count,xlab='Time',ylab='Rented Bicycle Count',geom="line")+theme_grey(base_size = 12) 
plot1
plot2 <-qplot(bike_data$Date[1:167],bike_data$Count[1:167],xlab='Time (1 week)',ylab='Rented Bicycle Count',geom="line")+theme_grey(base_size = 10) 
plot2

png('histogram_boxplot.png',width = 14, height = 10, units = 'in', res = 300)
par(mfrow=c(2,1))
hist(bike_data$Count,main="",xlab = "Rented Bike Count",breaks = 40,
     col='lightblue',xlim=c(0,4000),ylim=c(0,1400),cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)

boxplot(bike_data$Count,
        boxfill = "lightblue",horizontal=TRUE,ylim=c(0,4000),xlab="Rented Bike Count",frame=F,
        cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
dev.off()

pairs.panels(bike_data[2:11], 
             method = "kendall", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)

pairs.panels(bike_data[2:11], 
             method = "spearman", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)

pairs.panels(bike_data[2:11], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)
set.seed(1)
train_index <- createDataPartition(bike_data$Count,p=0.75,list=FALSE)
train_data <- bike_data[train_index,]
dim(train_data)
names(train_data)

# Saving training data in csv format

write.table(format(train_data, digits=19),file = "training.csv", sep = ",", row.names=FALSE)
# checkin if the two training (objects) files are the same afer saving
train_data2 <- read.csv("training.csv")
train_data2$Date <- strptime(as.character(train_data2$Date),format="%Y-%m-%d")
train_data2$Date <- as.POSIXct(train_data2$Date,tz = "UTC")
class(train_data2$Date)
str(train_data2)

#all(train_data ==train_data2) # TRUE
## they are the same

train_data <-train_data2

# creating the testing set here

test_data <- bike_data[-train_index,]
dim(test_data)
# 4932,32

write.table(format(test_data, digits=19),file = "testing.csv", sep = ",", row.names=FALSE)

testing2 <- read.csv("testing.csv")

testing2$Date <- strptime(as.character(testing2$Date),format="%Y-%m-%d")
testing2$Date <- as.POSIXct(testing2$Date,tz = "UTC")

names(test_data)
names(testing2)
# checking to see if the testings files (objects) are the same
all(testing2==test_data)
# TRUE
str(train_data)
head(train_data2)
str(train_data2)
new_train_data <- dummy.data.frame(train_data2,names=c("Wstatus","Dweek","Seasons","Holiday","Fday"))
str(new_train_data)
dim(new_train_data)
names(new_train_data)

write.table(format(new_train_data, digits=19),file = "testing.csv", sep = ",", row.names=FALSE)
write.table(format(new_test_data, digits=19),file = "testing.csv", sep = ",", row.names=FALSE)

names(new_train_data)[12] <- "Autumn"
names(new_train_data)[13] <- "Spring"
names(new_train_data)[14] <- "Summer"
names(new_train_data)[15] <- "Winter"
names(new_train_data)[16] <- "Holiday"
names(new_train_data)[17] <- "Workday"
names(new_train_data)[18] <- "NoFunc"
names(new_train_data)[19] <- "Func"
names(new_train_data)[20] <- "Fri"
names(new_train_data)[21] <- "Mon"
names(new_train_data)[22] <- "Sat"
names(new_train_data)[23] <- "Sun"
names(new_train_data)[24] <- "Thu"
names(new_train_data)[25] <- "Tue"
names(new_train_data)[26] <- "Wed"
names(new_train_data)[27] <- "Wday"
names(new_train_data)[28] <- "Wend"

new_test_data <- dummy.data.frame(test_data,names=c("Wstatus","Dweek" ,"Seasons","Holiday","Fday"))
str(new_test_data)
dim(new_test_data)
names(new_test_data)
names(new_test_data)[12] <- "Autumn"
names(new_test_data)[13] <- "Spring"
names(new_test_data)[14] <- "Summer"
names(new_test_data)[15] <- "Winter"
names(new_test_data)[16] <- "Holiday"
names(new_test_data)[17] <- "Workday"
names(new_test_data)[18] <- "NoFunc"
names(new_test_data)[19] <- "Func"
names(new_test_data)[20] <- "Fri"
names(new_test_data)[21] <- "Mon"
names(new_test_data)[22] <- "Sat"
names(new_test_data)[23] <- "Sun"
names(new_test_data)[24] <- "Thu"
names(new_test_data)[25] <- "Tue"
names(new_test_data)[26] <- "Wed"
names(new_test_data)[27] <- "Wday"
names(new_test_data)[28] <- "Wend"

write.table(format(new_train_data, digits=19),file = "new_train_data.csv", sep = ",", row.names=FALSE)
write.table(format(new_test_data, digits=19),file = "new_test_data.csv", sep = ",", row.names=FALSE)


gbmGrid <-  expand.grid(interaction.depth = c(1,3,12),
                        n.trees = seq(1,320,20),
                        shrinkage = 0.1,
                        n.minobsinnode = c(10))

set.seed(1)
registerDoParallel(4)
getDoParWorkers()
ptm <- proc.time()
gbm_model <- train(Count~., data=new_train_data[,c(2:28)],  method="gbm",
                   metric='Rsquared',trControl = fitControl,bag.fraction=0.5,tuneGrid=gbmGrid)
gbm_time <- proc.time() - ptm
gbm_model$bestTune
plot(gbm_model)
.rs.restartR()


lmcvFit <- train(Count~., data=new_train_data[,c(2:28)],  method="lm",trControl = fitControl,
                 metric='Rsquared')
lmcvFit$bestTune
residuals<-resid(lmcvFit)
predictedValues<-predict(lmcvFit)
plot(new_train_data$Count,residuals,xlab="Rented Bike Count", ylab="Residuals")
abline(0,0)
png('lm_model.png',width = 8, height = 6, units = 'in', res = 300)
plot(new_train_data$Count,residuals,xlab="Rented Bike Count", ylab="Residuals")
summary(lmcvFit)
plot(varImp(lmcvFit))
plot(lmcvFit)
plot(new_train_data$Count,predictedValues)
.rs.restartR()

set.seed(1)
registerDoParallel(4)
getDoParWorkers()
ptm <- proc.time()
rf_default <- train(Count~., data=new_train_data[,c(2:28)],
                    method="rf",metric='Rsquared',
                    trControl=fitControl,importance = TRUE )
rf_default_time <- proc.time() - ptm
rf_default_time
plot(rf_default)
rf_default$bestTune


.rs.restartR()

set.seed(1)
registerDoParallel(10)
getDoParWorkers()
ptm <- proc.time()
rf_mtry <- train(Count~., data=new_train_data[,c(2:28)],
                    method="rf",metric='Rsquared',
                    trControl=fitControl,importance = TRUE, tuneGrid=expand.grid(mtry=c(1:26)) )
rf_default_time <- proc.time() - ptm
rf_default_time
plot(rf_mtry)

rf_mtry$bestTune

.rs.restartR()

grid <- expand.grid(sigma = c(0.001,0.01,0.1),
                    C = c(1,10,20,30,40,50,60,70)
)

set.seed(1)
registerDoParallel(4)
getDoParWorkers()

ptm <- proc.time()
svm_model <- train(Count~., data=new_train_data[,c(2:28)],  method="svmRadial",
                   metric='Rsquared',trControl = fitControl, preProc=c("center","scale"),
                   tuneGrid = grid)
svm_time <- proc.time() - ptm
print(svm_model)
plot(svm_model)
.rs.restartR()


xgboostGrid <-  expand.grid(max_depth=c(1,2,3),
                            nrounds=c(10,100,200,300,400,500,600,700,800,900,1000,1100),
                            eta=c(0.1),
                            gamma = c(0),
                            colsample_bytree=1,
                            min_child_weight=1,
                            subsample=1)

set.seed(1)
registerDoParallel(4)
getDoParWorkers()
ptm <- proc.time()
xgb_model <- train(Count~., data=new_train_data[,c(2:28)],  method="xgbTree",
                   metric="Rsquared",trControl = fitControl,tuneGrid=xgboostGrid)
xgb_time <- proc.time() - ptm
xgb_model$call
xgb_model$bestTune
plot(xgb_model)
.rs.restartR()

btGrid <-  expand.grid(maxdepth=c(7,8,10),
                       mstop=seq(1,100,20) )

set.seed(1)
registerDoParallel(4)
getDoParWorkers()
ptm <- proc.time()
bt_model <- train(Count~., data=new_train_data[,c(2:28)],  method="blackboost",
                  metric='Rsquared',trControl = fitControl,tuneGrid=btGrid)
bt_time <- proc.time() - ptm
bt_model$call
bt_model$bestTune
plot(bt_model)

set.seed(1)
registerDoParallel(4)
getDoParWorkers()
ptm <- proc.time()
conditional_rf <- train(Count~., data=new_train_data[,c(2:28)],  method="cforest",
                        metric='Rsquared',trControl = fitControl)
conditional_rf_time <- proc.time() - ptm
print(conditional_rf)
plot(conditional_rf)
.rs.restartR()



set.seed(1)
registerDoParallel(4)
getDoParWorkers()
ptm <- proc.time()
rf_rand <- train(Count~., data=new_train_data[,c(2:28)],
                 method="extraTrees",metric='Rsquared',
                 trControl=fitControl)

rf_rand_time <- proc.time() - ptm
rf_rand_time
plot(rf_rand)
.rs.restartR()







set.seed(1)
registerDoParallel(4)
getDoParWorkers()

ptm <- proc.time()

rf_quantile <- train(Count~., data=new_train_data[,c(2:28)],
                     method="qrf",metric='Rsquared',
                     trControl=fitControl,importance = TRUE )

rf_quantile_time <- proc.time() - ptm
rf_quantile_time
plot(rf_quantile)



set.seed(1)
registerDoParallel(4)
getDoParWorkers()
ptm <- proc.time()
rrf_rf<- train(Count~., data=new_train_data[,c(2:28)],
               method="RRFglobal",metric='Rsquared',
               trControl=fitControl,importance = TRUE )
rrf_rf_time <- proc.time() - ptm
rrf_rf_time
rrf_rf$bestTune

plot(rrf_rf)

set.seed(1)
registerDoParallel(4)
getDoParWorkers()
ptm <- proc.time()
par_rf <- train(Count~., data=new_train_data[,c(2:28)],
                method="parRF",metric='Rsquared',
                trControl=fitControl,importance = TRUE )

par_rf_time <- proc.time() - ptm
par_rf
plot(par_rf)
save.image()

knngrid<- expand.grid(k=c(1:26))

set.seed(1)
registerDoParallel(4)
getDoParWorkers()
ptm <- proc.time()
knn <- train(Count~., data=new_train_data[,c(2:28)],  method="knn",
             metric='Rsquared',trControl = fitControl,preProc = c("center", "scale"),tuneGrid=knngrid)
knn_time <- proc.time() - ptm
knn_time
print(knn)
plot(knn)

cartgrid<- expand.grid(cp=c(0.0001,0.0001,0.001,0.01,0.02,0.04,0.06,0.08))
set.seed(1)
registerDoParallel(4)
getDoParWorkers()
ptm <- proc.time()
cart <- train(Count~., data=new_train_data[,c(2:28)],  method="rpart",
              metric='Rsquared',trControl = fitControl,tuneGrid=cartgrid)
cart_time <- proc.time() - ptm
cart_time
print(cart)
plot(cart)



params_ridge <- expand.grid(alpha=1,lambda=c(1,0.01,1,0.01,0.02,0))
set.seed(1)
registerDoParallel(4)
getDoParWorkers()
ptm <- proc.time()
ridge <- train(Count~., data=new_train_data[,c(2:28)],  method="glmnet",
               metric='Rsquared',trControl = fitControl,tuneGrid=params_ridge)
ridge_time <- proc.time() - ptm
ridge_time
print(ridge)
plot(ridge)


cubistGrid <-  expand.grid(committees = seq(1,50,5),
                           neighbors = c(1,3,5,7))
set.seed(1)
registerDoParallel(4)
getDoParWorkers()
ptm <- proc.time()
cubist <- train(Count~., data=new_train_data[,c(2:28)],  method="cubist",
                metric='Rsquared',trControl = fitControl,tuneGrid=cubistGrid)
cubist_time <- proc.time() - ptm
cubist_time
plot(cubist)
.rs.restartR()

Conditionalgrid<-expand.grid(mincriterion=c(0.001,0.01,0.1),
                             maxdepth=c(9,10,11,12,13,14,15,16,17,18,19,20,21))

set.seed(1)
registerDoParallel(4)
getDoParWorkers()
ptm <- proc.time()
conditional <- train(Count~., data=new_train_data[,c(2:28)],  method="ctree2",
                     metric='Rsquared',trControl = fitControl,tuneGrid= Conditionalgrid)
conditional_time <- proc.time() - ptm
conditional_time
plot(conditional)

set.seed(1)
registerDoParallel(4)
getDoParWorkers()
ptm <- proc.time()
gam <- train(Count~., data=new_train_data[,c(2:28)],  method="gamSpline",
             metric='Rsquared',trControl = fitControl,tuneGrid= expand.grid(df=c(3:24)))
gam_time <- proc.time() - ptm
gam_time
plot(gam)
             
set.seed(1)
registerDoParallel(4)
getDoParWorkers()
ptm <- proc.time()
svminear <- train(Count~., data=new_train_data[,c(2:28)],  method="svmLinear2",
                  metric='Rsquared',trControl = fitControl,
                  tuneGrid= expand.grid(cost=c(0.01,0.02,0.03,0.04,0.05)))
svmlinear_time <- proc.time() - ptm
svmlinear_time
plot(svminear)

.rs.restartR()
 

svmpolyGrid=expand.grid(degree=c(1,2,3,4,5,6,7),
                        scale=c(0.001,0.01,0.1),
                        C=1)                 
set.seed(1)
registerDoParallel(4)
getDoParWorkers()
ptm <- proc.time()
svm_poly <- train(Count~., data=new_train_data[,c(2:28)],  method="svmPoly",
                   metric='Rsquared',trControl = fitControl, preProc=c("center","scale"),tuneGrid=svmpolyGrid)
svm_time <- proc.time() - ptm
print(svm_poly)
plot(svm_poly)



svmboundaryGrid=expand.grid(length=c(1:4),
                        C=1)                
set.seed(1)
registerDoParallel(4)
getDoParWorkers()
ptm <- proc.time()
svm_spec <- train(Count~., data=new_train_data[,c(2:28)],  method="svmExpoString", 
                   tuneGrid=svmboundaryGrid)
svm_boundary_time <- proc.time() - ptm
print(svm_boundary)
plot(svm_boundary)

plot(new_test_data[,c(2:28)]$Count,predict(svm_model,new_test_data[,c(2:28)]))


Random Forest Results


Rsqu(new_train_data[,c(2:28)]$Count,predict(rf_default,new_train_data[,c(2:28)]))
rmse(new_train_data[,c(2:28)]$Count,predict(rf_default,new_train_data[,c(2:28)]))
MAE(new_train_data[,c(2:28)]$Count,predict(rf_default,new_train_data[,c(2:28)]))
cv(new_train_data[,c(2:28)]$Count,predict(rf_default,new_train_data[,c(2:28)]))*100



Rsqu(new_test_data[,c(2:28)]$Count,predict(rf_default,new_test_data[,c(2:28)]))
rmse(new_test_data[,c(2:28)]$Count,predict(rf_default,new_test_data[,c(2:28)]))
MAE(new_test_data[,c(2:28)]$Count,predict(rf_default,new_test_data[,c(2:28)]))
cv(new_test_data[,c(2:28)]$Count,predict(rf_default,new_test_data[,c(2:28)]))*100

plot(varImp(rf_default))
library(gridBase)
library(gridExtra)
library(lattice)
x <-1:500
y <- sqrt(rf_default$finalModel$mse)
n_trees <- xyplot(y ~ x, 
                  ylab="RMSE", xlab="Number of Trees")
dev.off()
rf_g <-plot(1:500,sqrt(rf_default$finalModel$mse),type='l',col='blue',axes=TRUE,xlab="Number of Trees",ylab="RMSE")

tress_g <- plot(rf_default)

class(tress_g)

panel_plot <- grid.arrange(n_trees, tress_g,ncol=2)
ggsave(file="rf_model.png", panel_plot)

cubist$bestTune


Rsqu(new_train_data[,c(2:28)]$Count,predict(cubist,new_train_data[,c(2:28)]))
rmse(new_train_data[,c(2:28)]$Count,predict(cubist,new_train_data[,c(2:28)]))
MAE(new_train_data[,c(2:28)]$Count,predict(cubist,new_train_data[,c(2:28)]))
cv(new_train_data[,c(2:28)]$Count,predict(cubist,new_train_data[,c(2:28)]))*100



Rsqu(new_test_data[,c(2:28)]$Count,predict(cubist,new_test_data[,c(2:28)]))
rmse(new_test_data[,c(2:28)]$Count,predict(cubist,new_test_data[,c(2:28)]))
MAE(new_test_data[,c(2:28)]$Count,predict(cubist,new_test_data[,c(2:28)]))
cv(new_test_data[,c(2:28)]$Count,predict(cubist,new_test_data[,c(2:28)]))*100

plot(varImp(cubist))
plot(cubist)
cubist$bestTune

Classification and Regression Trees

Rsqu(new_train_data[,c(2:28)]$Count,predict(cart,new_train_data[,c(2:28)]))
rmse(new_train_data[,c(2:28)]$Count,predict(cart,new_train_data[,c(2:28)]))
MAE(new_train_data[,c(2:28)]$Count,predict(cart,new_train_data[,c(2:28)]))
cv(new_train_data[,c(2:28)]$Count,predict(cart,new_train_data[,c(2:28)]))*100


Rsqu(new_test_data[,c(2:28)]$Count,predict(cart,new_test_data[,c(2:28)]))
rmse(new_test_data[,c(2:28)]$Count,predict(cart,new_test_data[,c(2:28)]))
MAE(new_test_data[,c(2:28)]$Count,predict(cart,new_test_data[,c(2:28)]))
cv(new_test_data[,c(2:28)]$Count,predict(cart,new_test_data[,c(2:28)]))*100

plot(varImp(cart))
plot(cart)
cart$bestTune

K Nearest Neighbour

Rsqu(new_train_data[,c(2:28)]$Count,predict(knn,new_train_data[,c(2:28)]))
rmse(new_train_data[,c(2:28)]$Count,predict(knn,new_train_data[,c(2:28)]))
MAE(new_train_data[,c(2:28)]$Count,predict(knn,new_train_data[,c(2:28)]))
cv(new_train_data[,c(2:28)]$Count,predict(knn,new_train_data[,c(2:28)]))*100


Rsqu(new_test_data[,c(2:28)]$Count,predict(knn,new_test_data[,c(2:28)]))
rmse(new_test_data[,c(2:28)]$Count,predict(knn,new_test_data[,c(2:28)]))
MAE(new_test_data[,c(2:28)]$Count,predict(knn,new_test_data[,c(2:28)]))
cv(new_test_data[,c(2:28)]$Count,predict(knn,new_test_data[,c(2:28)]))*100

plot(varImp(knn))
plot(knn)
knn$bestTune


Conditional Inference Tree

Rsqu(new_train_data[,c(2:28)]$Count,predict(conditional,new_train_data[,c(2:28)]))
rmse(new_train_data[,c(2:28)]$Count,predict(conditional,new_train_data[,c(2:28)]))
MAE(new_train_data[,c(2:28)]$Count,predict(conditional,new_train_data[,c(2:28)]))
cv(new_train_data[,c(2:28)]$Count,predict(conditional,new_train_data[,c(2:28)]))*100


Rsqu(new_test_data[,c(2:28)]$Count,predict(conditional,new_test_data[,c(2:28)]))
rmse(new_test_data[,c(2:28)]$Count,predict(conditional,new_test_data[,c(2:28)]))
MAE(new_test_data[,c(2:28)]$Count,predict(conditional,new_test_data[,c(2:28)]))
cv(new_test_data[,c(2:28)]$Count,predict(conditional,new_test_data[,c(2:28)]))*100

plot(varImp(conditional))
plot(conditional)
conditional$bestTune


RF_imp   <- plot(varImp(rf_default),main="RF Variable Importance")
CUBIST_imp  <- plot(varImp(cubist),main="CUBIST Variable Importance")
CART_imp  <- plot(varImp(cart),main="CART Variable Importance")
KNN_imp   <- plot(varImp(knn),main="KNN Variable Importance")
Conditional_imp  <- plot(varImp(conditional),main="CIT Variable Importance")



panelVIMP_plot_models <- grid.arrange(CUBIST_imp,RF_imp, CART_imp, KNN_imp,Conditional_imp,ncol=5)

ggsave(file="panel_plot_Cubist.png", panelVIMP_plot_models)


panelVIMP_plot_models <- grid.arrange(RF_imp, gbm_imp,SVM_imp,ncol=3)

ggsave(file="panel_plot_VIMP2.png", panelVIMP_plot_models)


rvalues <- resamples(list(CUBIST=cubist,RF=rf_default,CART=cart,KNN=knn,
                          CIT=conditional))

rvalues$values

RMSE_ALL <- dotplot(rvalues,metric = "RMSE")


RSQ_ALL <- dotplot(rvalues,metric="Rsquared")

MAE_ALL <- dotplot(rvalues,metric="MAE")


panel_plot_models <- grid.arrange(RSQ_ALL,RMSE_ALL,MAE_ALL, ncol=3)
panel_plot_models

ggsave(file="panel_plot_models2.png", panel_plot_models)








Rsqu(new_train_data[,c(2:28)]$Count,predict(cart,new_train_data[,c(2:28)]))
rmse(new_train_data[,c(2:28)]$Count,predict(cart,new_train_data[,c(2:28)]))
MAE(new_train_data[,c(2:28)]$Count,predict(cart,new_train_data[,c(2:28)]))
cv(new_train_data[,c(2:28)]$Count,predict(cart,new_train_data[,c(2:28)]))*100


Rsqu(new_test_data[,c(2:28)]$Count,predict(cart,new_test_data[,c(2:28)]))
rmse(new_test_data[,c(2:28)]$Count,predict(cart,new_test_data[,c(2:28)]))
MAE(new_test_data[,c(2:28)]$Count,predict(cart,new_test_data[,c(2:28)]))
cv(new_test_data[,c(2:28)]$Count,predict(cart,new_test_data[,c(2:28)]))*100

plot(varImp(cart))
plot(cart)
cart$bestTune




rmse(new_train_data[,c(2:28)]$Count,predict(svm_model,new_train_data[,c(2:28)]))
MAE(new_train_data[,c(2:28)]$Count,predict(svm_model,new_train_data[,c(2:28)]))
MAPE(new_train_data[,c(2:28)]$Count,predict(svm_model,new_train_data[,c(2:28)]))*100
Rsqu(new_train_data[,c(2:28)]$Count,predict(svm_model,new_train_data[,c(2:28)]))
mbe(new_train_data[,c(2:28)]$Count,predict(svm_model,new_train_data[,c(2:28)]))*100
cv(new_train_data[,c(2:28)]$Count,predict(svm_model,new_train_data[,c(2:28)]))*100

rmse(new_test_data[,c(2:28)]$Count,predict(svm_model,new_test_data[,c(2:28)]))
MAE(new_test_data[,c(2:28)]$Count,predict(svm_model,new_test_data[,c(2:28)]))
MAPE(new_test_data[,c(2:28)]$Count,predict(svm_model,new_test_data[,c(2:28)]))*100
Rsqu(new_test_data[,c(2:28)]$Count,predict(svm_model,new_test_data[,c(2:28)]))
mbe(new_test_data[,c(2:28)]$Count,predict(svm_model,new_test_data[,c(2:28)]))*100
cv(new_test_data[,c(2:28)]$Count,predict(svm_model,new_test_data[,c(2:28)]))*100


rmse(new_train_data[,c(2:28)]$Count,predict(svminear,new_train_data[,c(2:28)]))
MAE(new_train_data[,c(2:28)]$Count,predict(svminear,new_train_data[,c(2:28)]))
MAPE(new_train_data[,c(2:28)]$Count,predict(svminear,new_train_data[,c(2:28)]))*100
Rsqu(new_train_data[,c(2:28)]$Count,predict(svminear,new_train_data[,c(2:28)]))
mbe(new_train_data[,c(2:28)]$Count,predict(svminear,new_train_data[,c(2:28)]))*100
cv(new_train_data[,c(2:28)]$Count,predict(svminear,new_train_data[,c(2:28)]))*100

rmse(new_test_data[,c(2:28)]$Count,predict(svminear,new_test_data[,c(2:28)]))
MAE(new_test_data[,c(2:28)]$Count,predict(svminear,new_test_data[,c(2:28)]))
MAPE(new_test_data[,c(2:28)]$Count,predict(svminear,new_test_data[,c(2:28)]))*100
Rsqu(new_test_data[,c(2:28)]$Count,predict(svminear,new_test_data[,c(2:28)]))
mbe(new_test_data[,c(2:28)]$Count,predict(svminear,new_test_data[,c(2:28)]))*100
cv(new_test_data[,c(2:28)]$Count,predict(svminear,new_test_data[,c(2:28)]))*100


rmse(new_train_data[,c(2:28)]$Count,predict(svm_poly,new_train_data[,c(2:28)]))
MAE(new_train_data[,c(2:28)]$Count,predict(svm_poly,new_train_data[,c(2:28)]))
MAPE(new_train_data[,c(2:28)]$Count,predict(svm_poly,new_train_data[,c(2:28)]))*100
Rsqu(new_train_data[,c(2:28)]$Count,predict(svm_poly,new_train_data[,c(2:28)]))
mbe(new_train_data[,c(2:28)]$Count,predict(svm_poly,new_train_data[,c(2:28)]))*100
cv(new_train_data[,c(2:28)]$Count,predict(svm_poly,new_train_data[,c(2:28)]))*100

rmse(new_test_data[,c(2:28)]$Count,predict(svm_poly,new_test_data[,c(2:28)]))
MAE(new_test_data[,c(2:28)]$Count,predict(svm_poly,new_test_data[,c(2:28)]))
MAPE(new_test_data[,c(2:28)]$Count,predict(svm_poly,new_test_data[,c(2:28)]))*100
Rsqu(new_test_data[,c(2:28)]$Count,predict(svm_poly,new_test_data[,c(2:28)]))
mbe(new_test_data[,c(2:28)]$Count,predict(svm_poly,new_test_data[,c(2:28)]))*100
cv(new_test_data[,c(2:28)]$Count,predict(svm_poly,new_test_data[,c(2:28)]))*100

svm_model$bestTune
plot(svm_model)
plot(svminear)
plot(svm_poly)

plot(varImp(svm_model))
plot(varImp(svminear))
plot(varImp(svm_poly))

rvalues <- resamples(list(SVM_Radial=svm_model,SVM_POLY=svm_poly,SVM_LINEAR=svminear))
RMSE_ALL <- dotplot(rvalues,metric = "RMSE")
RSQ_ALL <- dotplot(rvalues,metric="Rsquared")
MAE_ALL <- dotplot(rvalues,metric = "MAE")


RSQ_ALL <- dotplot(rvalues,metric="Rsquared")

panel_plot_models <- grid.arrange( RSQ_ALL,RMSE_ALL,MAE_ALL,ncol=3)
panel_plot_models


Rsqu(new_test_data$Count,predict(xgb_model,new_test_data[,c(2:28)]))

Rsqu(new_test_data$Count,predict(cubist,new_test_data[,c(2:28)]))
rmse(new_test_data$Count,predict(cubist,new_test_data[,c(2:28)]))
MAE2(new_test_data$Count,predict(cubist,new_test_data[,c(2:28)]))
cv(new_test_data$Count,predict(cubist,new_test_data[,c(2:28)]))


Rsqu(new_test_data$Count,predict(rrf_rf,new_test_data[,c(2:28)]))
rmse(new_test_data$Count,predict(rrf_rf,new_test_data[,c(2:28)]))
MAE2(new_test_data$Count,predict(rrf_rf,new_test_data[,c(2:28)]))
cv(new_test_data$Count,predict(rrf_rf,new_test_data[,c(2:28)]))


rmse(new_test_data$Count,((predict(cubist,new_test_data[,c(2:28)])+predict(rrf_rf,new_test_data[,c(2:28)]))/2))
MAE2(new_test_data$Count,((predict(cubist,new_test_data[,c(2:28)])+predict(rrf_rf,new_test_data[,c(2:28)]))/2))
cv(new_test_data$Count,((predict(cubist,new_test_data[,c(2:28)])+predict(rrf_rf,new_test_data[,c(2:28)]))/2))
Rsqu(new_test_data$Count,((predict(cubist,new_test_data[,c(2:28)])+ predict(rrf_rf,new_test_data[,c(2:28)]))/2))



rmse(new_train_data[,c(2:28)]$Count,predict(knn,new_train_data[,c(2:28)]))
MAE(new_train_data[,c(2:28)]$Count,predict(knn,new_train_data[,c(2:28)]))
MAPE(new_train_data[,c(2:28)]$Count,predict(knn,new_train_data[,c(2:28)]))*100
Rsqu(new_train_data[,c(2:28)]$Count,predict(knn,new_train_data[,c(2:28)]))
mbe(new_train_data[,c(2:28)]$Count,predict(knn,new_train_data[,c(2:28)]))*100
cv(new_train_data[,c(2:28)]$Count,predict(knn,new_train_data[,c(2:28)]))*100

rmse(new_test_data[,c(2:28)]$Count,predict(knn,new_test_data[,c(2:28)]))
MAE(new_test_data[,c(2:28)]$Count,predict(knn,new_test_data[,c(2:28)]))
MAPE(new_test_data[,c(2:28)]$Count,predict(knn,new_test_data[,c(2:28)]))*100
Rsqu(new_test_data[,c(2:28)]$Count,predict(knn,new_test_data[,c(2:28)]))
mbe(new_test_data[,c(2:28)]$Count,predict(knn,new_test_data[,c(2:28)]))*100
cv(new_test_data[,c(2:28)]$Count,predict(knn,new_test_data[,c(2:28)]))*100



rmse(new_train_data[,c(2:28)]$Count,predict(cart,new_train_data[,c(2:28)]))
MAE(new_train_data[,c(2:28)]$Count,predict(cart,new_train_data[,c(2:28)]))
MAPE(new_train_data[,c(2:28)]$Count,predict(cart,new_train_data[,c(2:28)]))*100
Rsqu(new_train_data[,c(2:28)]$Count,predict(cart,new_train_data[,c(2:28)]))
mbe(new_train_data[,c(2:28)]$Count,predict(cart,new_train_data[,c(2:28)]))*100
cv(new_train_data[,c(2:28)]$Count,predict(cart,new_train_data[,c(2:28)]))*100

rmse(new_test_data[,c(2:28)]$Count,predict(cart,new_test_data[,c(2:28)]))
MAE(new_test_data[,c(2:28)]$Count,predict(cart,new_test_data[,c(2:28)]))
MAPE(new_test_data[,c(2:28)]$Count,predict(cart,new_test_data[,c(2:28)]))*100
Rsqu(new_test_data[,c(2:28)]$Count,predict(cart,new_test_data[,c(2:28)]))
mbe(new_test_data[,c(2:28)]$Count,predict(cart,new_test_data[,c(2:28)]))*100
cv(new_test_data[,c(2:28)]$Count,predict(cart,new_test_data[,c(2:28)]))*100



rmse(new_train_data[,c(2:28)]$Count,predict(rf_default,new_train_data[,c(2:28)]))
MAE(new_train_data[,c(2:28)]$Count,predict(rf_default,new_train_data[,c(2:28)]))
MAPE(new_train_data[,c(2:28)]$Count,predict(rf_default,new_train_data[,c(2:28)]))*100
Rsqu(new_train_data[,c(2:28)]$Count,predict(rf_default,new_train_data[,c(2:28)]))
mbe(new_train_data[,c(2:28)]$Count,predict(rf_default,new_train_data[,c(2:28)]))*100
cv(new_train_data[,c(2:28)]$Count,predict(rf_default,new_train_data[,c(2:28)]))*100

rmse(new_test_data[,c(2:28)]$Count,predict(rf_default,new_test_data[,c(2:28)]))
MAE(new_test_data[,c(2:28)]$Count,predict(rf_default,new_test_data[,c(2:28)]))
MAPE(new_test_data[,c(2:28)]$Count,predict(rf_default,new_test_data[,c(2:28)]))*100
Rsqu(new_test_data[,c(2:28)]$Count,predict(rf_default,new_test_data[,c(2:28)]))
mbe(new_test_data[,c(2:28)]$Count,predict(rf_default,new_test_data[,c(2:28)]))*100
cv(new_test_data[,c(2:28)]$Count,predict(rf_default,new_test_data[,c(2:28)]))*100







glm <- train(Count~., data=new_train_data[,c(2:28)],  method="glm",trControl = fitControl,
                 metric='Rsquared')



Rsqu(new_train_data[,c(2:28)]$Count,predict(glm,new_train_data[,c(2:28)]))
rmse(new_train_data[,c(2:28)]$Count,predict(glm,new_train_data[,c(2:28)]))
MAE(new_train_data[,c(2:28)]$Count,predict(glm,new_train_data[,c(2:28)]))
cv(new_train_data[,c(2:28)]$Count,predict(glm,new_train_data[,c(2:28)]))*100


Rsqu(new_test_data[,c(2:28)]$Count,predict(glm,new_test_data[,c(2:28)]))
rmse(new_test_data[,c(2:28)]$Count,predict(glm,new_test_data[,c(2:28)]))
MAE(new_test_data[,c(2:28)]$Count,predict(glm,new_test_data[,c(2:28)]))
cv(new_test_data[,c(2:28)]$Count,predict(glm,new_test_data[,c(2:28)]))*100


Rsqu(new_train_data[,c(2:28)]$Count,predict(rrf_rf,new_train_data[,c(2:28)]))
rmse(new_train_data[,c(2:28)]$Count,predict(rrf_rf,new_train_data[,c(2:28)]))
MAE(new_train_data[,c(2:28)]$Count,predict(rrf_rf,new_train_data[,c(2:28)]))
cv(new_train_data[,c(2:28)]$Count,predict(rrf_rf,new_train_data[,c(2:28)]))*100


Rsqu(new_test_data[,c(2:28)]$Count,predict(rrf_rf,new_test_data[,c(2:28)]))
rmse(new_test_data[,c(2:28)]$Count,predict(rrf_rf,new_test_data[,c(2:28)]))
MAE(new_test_data[,c(2:28)]$Count,predict(rrf_rf,new_test_data[,c(2:28)]))
cv(new_test_data[,c(2:28)]$Count,predict(rrf_rf,new_test_data[,c(2:28)]))*100



Rsqu(new_train_data[,c(2:28)]$Count,predict(conditional,new_train_data[,c(2:28)]))
rmse(new_train_data[,c(2:28)]$Count,predict(conditional,new_train_data[,c(2:28)]))
MAE(new_train_data[,c(2:28)]$Count,predict(conditional,new_train_data[,c(2:28)]))
cv(new_train_data[,c(2:28)]$Count,predict(conditional,new_train_data[,c(2:28)]))*100


Rsqu(new_test_data[,c(2:28)]$Count,predict(conditional,new_test_data[,c(2:28)]))
rmse(new_test_data[,c(2:28)]$Count,predict(conditional,new_test_data[,c(2:28)]))
MAE(new_test_data[,c(2:28)]$Count,predict(conditional,new_test_data[,c(2:28)]))
cv(new_test_data[,c(2:28)]$Count,predict(conditional,new_test_data[,c(2:28)]))*100




residuals<-resid(glm)

predictedValues<-predict(glm)

plot(new_train_data$Count,residuals,xlab="Rental Bike Count", ylab="Residuals")

abline(0,0)

png('lm_model.png',width = 8, height = 6, units = 'in', res = 300)
plot(new_train_data$Count,residuals,xlab="Rental Bike Count", ylab="Residuals")
abline(0,0)
dev.off()

library(gridBase)
library(gridExtra)
library(lattice)
x <-1:500
y <- sqrt(rrf_rf$finalModel$mse)
n_trees <- xyplot(y ~ x, 
                  ylab="RMSE", xlab="Number of Trees")
dev.off()
rf_g <-plot(1:500,sqrt(rrf_rf$finalModel$mse),type='l',col='blue',axes=TRUE,xlab="Number of Trees",ylab="RMSE")

tress_g <- plot(rrf_rf)

class(tress_g)

panel_plot <- grid.arrange(n_trees, tress_g,ncol=2)
ggsave(file="rrf_rf_model.png", panel_plot)


GLM_imp   <- plot(varImp(glm),main="GLM Variable Importance")
RRF_imp  <- plot(varImp(rrf_rf),main="RRF Variable Importance")
CIT_imp  <- plot(varImp(conditional),main="CIT Variable Importance")
CUBIST_imp  <- plot(varImp(cubist),main="CUBIST Variable Importance")

panelVIMP_plot_models <- grid.arrange(GLM_imp, RRF_imp,CIT_imp,CUBIST_imp,ncol=4)
ggsave(file="srmvarimp.png", panelVIMP_plot_models)





rvalues <- resamples(list(GLM=glm,RRF=rrf_rf,CIT=conditional,CUBIST=cubist))
RMSE_ALL <- dotplot(rvalues,metric = "RMSE")
RSQ_ALL <- dotplot(rvalues,metric="Rsquared")



panel_plot_models <- grid.arrange(RSQ_ALL, RMSE_ALL, ncol=2)
panel_plot_models
ggsave(file="panel_plot_srm.png", panel_plot_models)

grid.arrange(RSQ_ALL, RMSE_ALL,MAE,ncol=3)



plot(glm)
plot(rrf_rf)
plot(conditional)
plot(cubist)







