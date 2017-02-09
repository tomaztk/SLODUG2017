###
### SLO_DUG 09.02.2017
### Leaf Classification
### R - with MicrosoftML and RevoscaleR
###

# NOTE: Make sure you connect to R Engine: 64Bit - [64-bit] C:\Program Files\Microsoft SQL Server\140\R_SERVER
# SET: Tools -> Global Options
setwd("C:/DataTK/Kaggle/Leaf_Classification/")
# sessionInfo()
# dfip <- data.frame(installed.packages())
# unique(dfip$LibPath)
# .libPaths()
# rm(dfip)


# Load Packages
library(MicrosoftML)
library(RevoScaleR)
library(ggplot2)
library(readr)
library(magrittr)
library(dplyr)
library(corrplot)
library(MLmetrics)


# Read Data
train <- read.csv("C:/DataTK/Kaggle/Leaf_Classification/train.csv")
test <- read.csv("C:/DataTK/Kaggle/Leaf_Classification/test.csv")

#we don't need labels
#use it for predictions!
train<- train[,-1]

# check the class for the train dataset
sapply(train, class)


# Check for missing values!
Num_NA<-sapply(train,function(y)length(which(is.na(y)==T)))
sum(Num_NA)



# get data into XDF format
#readPath <- rxGetOption("sampleDataDir")
infile <- file.path("C:/DataTK/Kaggle/Leaf_Classification/", "train.csv")
trainDF <- rxImport(infile)
trainData <- rxImport(inData = trainDF, outFile="Leaf_train.xdf", 
                    stringsAsFactors = TRUE, missingValueString = "M", 
                    rowsPerRead = 200000, overwrite = TRUE)

#check traindata
# "Leaf_train.xdf"
# File name: C:\DataTK\Kaggle\Leaf_Classification\Leaf_train.xdf 
rxGetInfo(trainData, getVarInfo = TRUE)


#data preparation
str(trainData) #xdf data
str(train)     #data frame
names(trainData)

formula <- "species ~ margin1 + margin2 + margin3 + margin4 + margin5 + margin6 + margin7 + margin8 + margin9 + margin10 + margin11 + margin12 + margin13 + margin14
+ margin15 + margin16 + margin17 + margin18 + margin19 + margin20 + margin21 + margin22 + margin23 + margin24 + margin25 + margin26 + margin27 + margin28 + margin29 
+ margin30 + margin31 + margin32 + margin33 + margin34 + margin35 + margin36 + margin37 + margin38 + margin39 + margin40 + margin41 + margin42 + margin43 + margin44 
+ margin45 + margin46 + margin47 + margin48 + margin49 + margin50 + margin51 + margin52 + margin53 + margin54 + margin55 + margin56 + margin57 + margin58 + margin59 
+ margin60 + margin61 + margin62 + margin63 + margin64 + shape1 + shape2 + shape3 + shape4 + shape5 + shape6 + shape7 + shape8 + shape9 + shape10
+ shape11 + shape12 + shape13 + shape14 + shape15 + shape16 + shape17 + shape18 + shape19 + shape20 + shape21 + shape22 + shape23 + shape24 + shape25  
+ shape26 + shape27 + shape28 + shape29 + shape30 + shape31 + shape32 + shape33 + shape34 + shape35 + shape36 + shape37 + shape38 + shape39 + shape40  
+ shape41 + shape42 + shape43 + shape44 + shape45 + shape46 + shape47 + shape48 + shape49 + shape50 + shape51 + shape52 + shape53 + shape54 + shape55  
+ shape56 + shape57 + shape58 + shape59 + shape60 + shape61 + shape62 + shape63 + shape64 + texture1 + texture2 + texture3 + texture4 + texture5 + texture6 
+ texture7 + texture8 + texture9 + texture10 + texture11 + texture12 + texture13 + texture14 + texture15 + texture16 + texture17 + texture18 + texture19 + texture20 + texture21
+ texture22 + texture23 + texture24 + texture25 + texture26 + texture27 + texture28 + texture29 + texture30 + texture31 + texture32 + texture33 + texture34 + texture35 + texture36
+ texture37 + texture38 + texture39 + texture40 + texture41 + texture42 + texture43 + texture44 + texture45 + texture46 + texture47 + texture48 + texture49 + texture50 + texture51
+ texture52 + texture53 + texture54 + texture55 + texture56 + texture57 + texture58 + texture59 + texture60 + texture61 + texture62 + texture63 + texture64"


scope <- "margin1 + margin2 + margin3 + margin4 + margin5 + margin6 + margin7 + margin8 + margin9 + margin10 + margin11 + margin12 + margin13 + margin14
+ margin15 + margin16 + margin17 + margin18 + margin19 + margin20 + margin21 + margin22 + margin23 + margin24 + margin25 + margin26 + margin27 + margin28 + margin29 
+ margin30 + margin31 + margin32 + margin33 + margin34 + margin35 + margin36 + margin37 + margin38 + margin39 + margin40 + margin41 + margin42 + margin43 + margin44 
+ margin45 + margin46 + margin47 + margin48 + margin49 + margin50 + margin51 + margin52 + margin53 + margin54 + margin55 + margin56 + margin57 + margin58 + margin59 
+ margin60 + margin61 + margin62 + margin63 + margin64 + shape1 + shape2 + shape3 + shape4 + shape5 + shape6 + shape7 + shape8 + shape9 + shape10
+ shape11 + shape12 + shape13 + shape14 + shape15 + shape16 + shape17 + shape18 + shape19 + shape20 + shape21 + shape22 + shape23 + shape24 + shape25  
+ shape26 + shape27 + shape28 + shape29 + shape30 + shape31 + shape32 + shape33 + shape34 + shape35 + shape36 + shape37 + shape38 + shape39 + shape40  
+ shape41 + shape42 + shape43 + shape44 + shape45 + shape46 + shape47 + shape48 + shape49 + shape50 + shape51 + shape52 + shape53 + shape54 + shape55  
+ shape56 + shape57 + shape58 + shape59 + shape60 + shape61 + shape62 + shape63 + shape64 + texture1 + texture2 + texture3 + texture4 + texture5 + texture6 
+ texture7 + texture8 + texture9 + texture10 + texture11 + texture12 + texture13 + texture14 + texture15 + texture16 + texture17 + texture18 + texture19 + texture20 + texture21
+ texture22 + texture23 + texture24 + texture25 + texture26 + texture27 + texture28 + texture29 + texture30 + texture31 + texture32 + texture33 + texture34 + texture35 + texture36
+ texture37 + texture38 + texture39 + texture40 + texture41 + texture42 + texture43 + texture44 + texture45 + texture46 + texture47 + texture48 + texture49 + texture50 + texture51
+ texture52 + texture53 + texture54 + texture55 + texture56 + texture57 + texture58 + texture59 + texture60 + texture61 + texture62 + texture63 + texture64"



#-------------------------------------------------------
##Classification and Regression Decision Trees (rxDTree)
#-------------------------------------------------------

#start time
ptm <- proc.time()

LC_CRT <- rxDTree(formula, data = train, maxDepth = 5, cp=0.01, xval = 0, blocksPerRead = 200)
pred_CRT <- rxPredict(LC_CRT,data=train, type='prob') 
logloss_CRT<-MultiLogLoss(y_true = train$species, y_pred = as.matrix(pred_CRT))

#end time
time_CRT <- proc.time() - ptm	
time_CRT <- time_CRT[[3]]





#---------------------------------------------
##Stochastic Gradient Boosting  (rxBTrees)
#---------------------------------------------

# DON'T RUN AT PRESENTATION
#start time
ptm <- proc.time()

LC_GBT <- rxBTrees(formula, data = trainData, maxDepth = 5, nTree = 50, lossFunction = "multinomial")
pred_GBT <- rxPredict(LC_GBT,data=train, type='prob') 
logloss_GBT<-MultiLogLoss(y_true = train$species, y_pred = as.matrix(pred_GBT))

#end time
time_GBT <- proc.time() - ptm	

logloss_GBT <- 2.6467047565207934
time_GBT <- 3405.21

#-----------------------------------------------------------
##Classification and Regression Decision Forests (rxDForest)
#-----------------------------------------------------------

# runs cca 50sec
#start time
ptm <- proc.time()


LC_DF <- rxDForest(formula, data = train, maxDepth = 5, cp=0.01, xval = 0, blocksPerRead = 200)
pred_DF <- rxPredict(LC_DF,data=train, type='prob')   
logloss_DF<-MultiLogLoss(y_true = train$species, y_pred = as.matrix(pred_DF[,c(1:99)]))

#end time
time_DF <- proc.time() - ptm	
time_DF <-time_DF[[3]]

#-------------------------------------------------------------------------
# # MicrosoftML -  Multi-class logistic regression  (rxLogisticRegression)
#-------------------------------------------------------------------------

#start time
ptm <- proc.time()

LC_MCLR <- rxLogisticRegression(formula = formula, type = "multiClass", data = train)
pred_MCLR <- rxPredict(LC_MCLR,data=train)
logloss_MCLR<-MultiLogLoss(y_true = train$species, y_pred = as.matrix(pred_MCLR[,c(2:100)]))


#end time
time_MCLR <- proc.time() - ptm	
time_MCLR <- time_MCLR[[3]]


#-------------------------------------------------------------------------
# # MicrosoftML - # Multi-class  (regression) neural net (rxNeuralNet)
#-------------------------------------------------------------------------

#start time
ptm <- proc.time()


LC_MNN <- rxNeuralNet(formula = formula,  data = train, type = "multiClass")
pred_MNN <- rxPredict(LC_MNN, data = train, extraVarsToWrite = "species")
logloss_MNN <- MultiLogLoss(y_true = train$species, y_pred = as.matrix(pred_MNN[,c(3:101)]))

#end time
time_MNN <- proc.time() - ptm	
time_MNN <- time_MNN[[3]]



#-------------------------------------------------------------------------
# Algorithm LOG Loss comparison
#-------------------------------------------------------------------------
MLLoss <- data.frame(
           method=c('Classification Decision Trees','Gradient Boosting',
                    'Classification Decision Forests', 'Multi-class logistic regression', 'Multi-class Neural Net')
          ,Multilogloss=c(logloss_CRT,logloss_GBT, logloss_DF, logloss_MCLR,logloss_MNN)
          ,TrainTime = c(time_CRT,time_GBT,time_DF,time_MCLR,time_MNN)
          )



ggplot(MLLoss,aes(x=method,y=Multilogloss, fill=method))+
  geom_bar(stat='identity')+
  ggtitle('MultiLog Loss for Leaf classification model efficiency with RevoScaleR/MicrosoftML') 



MLLoss



#------------------ END OF FILE




# Make a submission

















