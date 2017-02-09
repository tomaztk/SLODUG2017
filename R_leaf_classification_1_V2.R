###
### SLO_DUG 09.02.2017
### Leaf Classification
### R - without MicrosoftML and RevoscaleR and using PCA for complexity reduction
###


# NOTE: Make sure you connect to R Engine: 64Bit - [64-bit] C:\Program Files\Microsoft SQL Server\130\R_SERVER
# Tools -> Global Options
setwd("C:/DataTK/Kaggle/Leaf_Classification/")

# Load Packages
library(ggplot2)
library(readr)
library(magrittr)
library(dplyr)
library(e1071)
library(rpart)
library(Metrics)
library(randomForest)
library(Matrix)
library(methods)
library(MLmetrics)
library(rpart.plot)
library(corrplot)
library(xgboost)
library(caret)



# Read Data
train <- read.csv("C:/DataTK/Kaggle/Leaf_Classification/train.csv")
test <- read.csv("C:/DataTK/Kaggle/Leaf_Classification/test.csv")

train<- train[,-1]
test<- test


sapply(train, class)


Num_NA<-sapply(train,function(y)length(which(is.na(y)==T)))
sum(Num_NA)



#-------------------------
# PCA! Reduction for each of three groups of attributes (Margin, Shape, texture)
#-------------------------
Margin<-train %>% select(contains("margin"))
pr_Margin<-princomp(Margin)
Shape<- train %>% select(contains('shape'))
pr_shape<-princomp(Shape)
Texture<- train %>% select(contains('texture'))
pr_texture<-princomp(Texture)


#Check the loadings
summary(pr_Margin)
summary(pr_shape)
summary(pr_texture)

#Take the best
Train_PCA<- data.frame(train$species,pr_Margin$scores[,1:5],pr_shape$scores[,1:3],pr_texture$scores[,1:5])
colnames(Train_PCA)<-c('species','Com1','Com2','Com3','Com4','Com5','Com6','Com7','Com8','Com9','Com10','Com11','Com12','Com13')
Test_Margin<- predict(pr_Margin,newdata=test %>% select(contains("margin")))[,1:5]
Test_Shape<- predict(pr_shape,newdata=test %>% select(contains("shape")))[,1:3]
Test_Texture<- predict(pr_texture,newdata=test %>%select(contains("texture")))[,1:5]
Test<- data.frame(Test_Margin,Test_Shape,Test_Texture)
colnames(Test)<-c('Com1','Com2','Com3','Com4','Com5','Com6','Com7','Com8','Com9','Com10','Com11','Com12','Com13')


#-------------------------------------------------------------------------
# # Naive Bayes - (1Sec)
#-------------------------------------------------------------------------

#start time
ptm <- proc.time()

NaivB<- naiveBayes(species~.,Train_PCA)
pred_NaivB <- predict(NaivB,newdata=Train_PCA[,2:14],type='raw')
logloss_NaivB <-MultiLogLoss(y_true = Train_PCA[,1], y_pred = as.matrix(pred_NaivB))

#end time
time_NaivB <- proc.time() - ptm	
time_NaivB <- time_NaivB[[3]]



#-------------------------------------------------------------------------
# classification tree - (2Sec)
#-------------------------------------------------------------------------

#start time
ptm <- proc.time()

Control<- trainControl(method='repeatedcv',number =10,repeats=3)
Tree<- train(Train_PCA[,2:14],Train_PCA[,1],method='rpart',trControl=Control)
pred_Tree<- predict(Tree,newdata= Train_PCA[,2:14],type='prob')
logloss_Tree<- MultiLogLoss(y_true = Train_PCA[,1], y_pred = as.matrix(pred_Tree))

#end time
time_Tree <- proc.time() - ptm	
time_Tree <- time_Tree[[3]]



#-------------------------------------------------------------------------
# # Random forest - (2min)
#-------------------------------------------------------------------------

#start time
ptm <- proc.time()


Control<- trainControl(method='repeatedcv',number =10,repeats=3)
rf<- train(Train_PCA[,2:14],Train_PCA[,1],method='rf',prox=TRUE,allowParallel=TRUE,trControl=Control)
pred_rf<- predict(rf,newdata= Train_PCA[,2:14],type='prob')
logloss_rf<-MultiLogLoss(y_true = Train_PCA[,1], y_pred = as.matrix(pred_rf))

#end time
time_rf <- proc.time() - ptm	
time_rf <- time_rf[[3]]


time_rf <-  185.95
logloss_rf <- 0.1972274
  
#-------------------------------------------------------------------------
# # Multinominal Logit Regression - (~ 5min)
#-------------------------------------------------------------------------

#start time
ptm <- proc.time()

Control<- trainControl(method='repeatedcv',number = 10,repeats=3)
Grid<- expand.grid(decay=c(0.0001,0.0000001,0.00000001))
LG<-train(Train_PCA[,2:14],Train_PCA[,1],method='multinom',prox=TRUE,allowParallel=TRUE,trControl=Control,tuneGrid=Grid,MaxNWts=2000)

# Running predictions for Logit Regression
pred_LG<- predict(LG,newdata= Train_PCA[,2:14],type='prob')
logloss_LG <-MultiLogLoss(y_true = Train_PCA[,1], y_pred = as.matrix(pred_LG))

#end time
time_LG <- proc.time() - ptm	
time_LG <- time_LG[[3]]

time_LG <- 280.52
logloss_LG <- 0.2146564

#-------------------------------------------------------------------------
# #XGboost - God please don't run this at live demo!!! - (30min - R Killer)!!!
#-------------------------------------------------------------------------


#start time
ptm <- proc.time()

cv.ctrl <- trainControl(method = "repeatedcv", repeats = 10,number = 3)

xgb.grid <- expand.grid(nrounds = 100,
                        max_depth = seq(6,10),
                        eta = c(0.01,0.3, 1),
                        gamma = c(0.0, 0.2, 1),
                        colsample_bytree = c(0.5,0.8, 1),
                        min_child_weight= 1
)

xgb_tune <-train(species ~.,
                 data=Train_PCA,
                 method="xgbTree",
                 trControl=cv.ctrl,
                 tuneGrid=xgb.grid
)

#predictions for XGboost
pred_xgb<- predict(xgb_tune,newdata= Train_PCA[,2:14],type='prob')
logloss_xgb<-MultiLogLoss(y_true = Train_PCA[,1], y_pred = as.matrix(pred_xgb))


#end time
time_xgb<- proc.time() - ptm	
time_xgb <- time_xgb[[3]]

# a Hack for SLODug Demo :)
logloss_xgb <- 0.404116
time_xgb  <- 18429  #300 minut 

# Comparison
MLL_Perf<- data.frame(method=c('NaiveBayes','Classification Tree','Random Forest','Logistic Regression','XGBoost'), 
                     Multilogloss=c(logloss_NaivB,logloss_Tree,logloss_rf,logloss_LG,logloss_xgb),
                     TrainTime = c(time_NaivB,time_Tree,time_rf,time_LG,time_xgb))

  
  ggplot(MLL_Perf,aes(x=method,y=Multilogloss, fill=method)) + 
                geom_bar(stat='identity') +
                ggtitle('MultiLog Loss for Leaf classification PCA model efficiency  with e1071/caret/xgboost/RandomForest')


# Check Times:
MLL_Perf
