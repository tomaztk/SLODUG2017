###
### SLO_DUG 09.02.2017
### Digit recognizer
### with MicrosoftML - Neural Net - DNN with Convolutions
### # Data available: https://www.kaggle.com/c/digit-recognizer/data


# NOTE: Make sure you connect to R Engine: 64Bit - [64-bit] C:\Program Files\Microsoft SQL Server\140\R_SERVER
# Tools -> Global Options
setwd("C:/DataTK/Kaggle/Digit_recognition/")

# Load Packages
library(MicrosoftML)
library(RevoScaleR)
library(ggplot2)
library(readr)
library(magrittr)
library(dplyr)
library(corrplot)



#----------------------------
## Check the data
#----------------------------


library(readr)
train <- read.csv("C:/DataTK/Kaggle/Digit_recognition/train.csv")
test <- read.csv("C:/DataTK/Kaggle/Digit_recognition/test.csv")
head(train[1:10])

# Create a 28*28 matrix with pixel color values
m = matrix(unlist(train[10,-1]),nrow = 28,byrow = T)
# Plot that matrix
image(m,col=grey.colors(255))

rotate <- function(x) t(apply(x, 2, rev)) # reverses (rotates the matrix)

# Plot a bunch of images
par(mfrow=c(2,3))
lapply(1:6, 
       function(x) image(
         rotate(matrix(unlist(train[x,-1]),nrow = 28,byrow = T)),
         col=grey.colors(255),
         xlab=train[x,1]
       )
)


#plot back
par(mfrow=c(1,1)) 





#------------------------------------
#
# MicrosoftML
#
#------------------------------------


# Read Data

train_DR <- read.csv("C:/DataTK/Kaggle/Digit_recognition/train.csv")
test_DR <- read.csv("C:/DataTK/Kaggle/Digit_recognition/test.csv")



# get data into XDF format
#readPath <- rxGetOption("sampleDataDir")
infile_train <- file.path("C:/DataTK/Kaggle/Digit_recognition/", "train.csv")
infile_test <- file.path("C:/DataTK/Kaggle/Digit_recognition/", "test.csv")

trainDR <- rxImport(infile_train)
trainData <- rxImport(inData = trainDR, outFile="Digit_train.xdf", 
                      stringsAsFactors = TRUE, missingValueString = "M", rowsPerRead = 200000, overwrite = TRUE)

testDR <- rxImport(infile_test)
testData <- rxImport(inData = testDR, outFile="Digit_test.xdf", 
                      stringsAsFactors = TRUE, missingValueString = "M", rowsPerRead = 200000, overwrite = TRUE)



#check traindata
# "Digit_train.xdf"
# "Digit_test.xdf"
# File name: 
rxGetInfo(trainDR, getVarInfo = TRUE) #has label variable called "label"
rxGetInfo(testDR, getVarInfo = TRUE)  #does not have label variable


dataTrain <- rxReadXdf("Digit_train.xdf")
dataTest <- rxReadXdf("Digit_test.xdf")




# NET# language for DDN
# More on NET# https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-azure-ml-netsharp-reference-guide
# or: https://blogs.technet.microsoft.com/machinelearning/2015/02/16/neural-nets-in-azure-ml-introduction-to-net/

netDefinition <- '
// Define constants.
const { T = true; F = false; }

// Input layer definition.
input Picture [28, 28];

// First convolutional layer definition.
hidden C1 [5 * 13^2]
from Picture convolve {
InputShape  = [28, 28];
UpperPad    = [ 1,  1];
KernelShape = [ 5,  5];
Stride      = [ 2,  2];
MapCount = 5;
}

// Second convolutional layer definition.
hidden C2 [50, 5, 5]
from C1 convolve {
InputShape  = [ 5, 13, 13];
KernelShape = [ 1,  5,  5];
Stride      = [ 1,  2,  2];
Sharing     = [ F,  T,  T];
MapCount = 10;
}

// Third fully connected layer definition.
hidden H3 [100]
from C2 all;

// Output layer definition.
output Result [10]
from H3 all;
'



# Train the neural Network

# Download CUDA Drivers: https://developer.nvidia.com/cuda-downloads

#start time
ptm <- proc.time()

#GPU
model_DNN_GPU <- rxNeuralNet(label ~.
                             ,data = dataTrain
                             ,type = "multi"
                             ,numIterations = 10
                             ,normalize = "no"
                             ,acceleration = "gpu" #enable this if you have CUDA driver
                             ,miniBatchSize = 64 #set to 1 else set to 64 if you have CUDA driver problem 
                             #,netDefinition = readChar(netDefFile, file.info(netDefFile)$size)
                             ,netDefinition = netDefinition
                             ,optimizer = sgd(learningRate = 0.1, lRateRedRatio = 0.9, lRateRedFreq = 10)
                             )

#end time
time_MSFTML_NN <- proc.time() - ptm	
time_MSFTML_NN <- time_MSFTML_NN[[3]]

# with GPU 
# time_MSFTML_NN <- 14.673564
time_MSFTML_NN <-  85.84
  
DNN_GPU_score <- rxPredict(model_DNN_GPU, dataTest, extraVarsToWrite = "label")
rxCrossTabs(formula = ~F(label):PredictLabel, data=DNN_GPU_score)

# Accuracy
sum(Score_DNN$Label == DNN_GPU_score$PredictedLabel)/dim(DNN_GPU_score)[1]


# ---------------
# without GPU
# ---------------

#netDefFile <- system.file("demoScripts/mnist.nn", package = "MicrosoftML")
#source(system.file("extdata/mnist.R", package = "MicrosoftML"))

mnist <- getMnistData(download = TRUE, sampleDataDir = NULL, createDir = TRUE)
mnistTrain <- mnist$mnistTrain
mnistTest <- mnist$mnistTest

#start time
ptm <- proc.time()


# multiClass with rxNeuralNet
Model_DNN <- rxNeuralNet(Label ~ .
                       ,data = mnistTrain
                       ,numIterations = 10
                       ,normalize = "no"
                       ,optimizer = sgd(learningRate=0.1, lRateRedRatio=0.9, lRateRedFreq=10)
                       #,netDefinition = readChar(netDefFile, file.info(netDefFile)$size)
                       ,netDefinition = netDefinition
                       ,type = "multi")

#end time
time_MSFTML_NN_NoGPU <- proc.time() - ptm	
time_MSFTML_NN_NoGPU <- time_MSFTML_NN_NoGPU[[3]]

time_MSFTML_NN_NoGPU <- 125.11

Score_DNN <- rxPredict(Model_DNN, mnistTest, extraVarsToWrite = "Label")


rxCrossTabs(formula = ~ F(Label):PredictedLabel, data = Score_DNN)

# Show the (micro-)accuracy
sum(Score_DNN$Label == Score_DNN$PredictedLabel)/dim(Score_DNN)[1]
# [1] 0.9767







#---------------------------------
## Using H20
#---------------------------------


library(h2o)
localH2O = h2o.init(max_mem_size = '4g', nthreads = -1) 


## MNIST data as H2O
#convert digit labels to factor for classification
train[,1] <- as.factor(train[,1]) 
train_h2o <- as.h2o(train)


test_h2o <- as.h2o(test)



#start time
ptm <- proc.time()


## train model
model_h20 <- h2o.deeplearning(x = 2:785
                         ,y = 1   # label for label
                        ,training_frame = train_h2o
                        ,activation = "RectifierWithDropout"
                        ,input_dropout_ratio = 0.2 # % of inputs dropout
                        ,hidden_dropout_ratios = c(0.5,0.5) # % for nodes dropout
                        ,balance_classes = TRUE 
                        ,hidden = c(100,100) # two layers of 100 nodes
                        ,momentum_stable = 0.99
                        ,nesterov_accelerated_gradient = T # use it for speed
                        ,epochs = 15) 

#end time
time_h20DL <- proc.time() - ptm	
time_h20DL <- time_h20DL[[3]]


time_h20DL <- 142.01

# success rate matrix
h2o.confusionMatrix(model_h20)

#exit the h20
h2o.shutdown()


# --------------------
# Comparison
# --------------------

Compare_NN<- data.frame(
                      method=c('ML_NN_GPU','ML_NN_Non_GPU','H2o','XGBoost'), 
                      TrainTime = c(time_MSFTML_NN,time_MSFTML_NN_NoGPU, time_h20DL, time_xgb_NN)
                      )




ggplot(Compare_NN,aes(x=method,y=TrainTime, fill=method))+
  geom_bar(stat='identity')+
  ggtitle('NeuralNetwork Digit recognizer train time comparison') 





# --------------------------------------
#   XgBoost
# ------------------------------------


library(readr)
library(ggplot2)
library(caret)
library(Matrix)
library(xgboost)


# data preparation
TRAIN <- read.csv("../input/train.csv")
TEST <- read.csv("../input/test.csv")
LABEL <- TRAIN$label
TRAIN$label <- NULL
LINCOMB <- findLinearCombos(TRAIN)
TRAIN <- TRAIN[, -LINCOMB$remove]
TEST <- TEST[, -LINCOMB$remove]
NZV <- nearZeroVar(TRAIN, saveMetrics = TRUE)
TRAIN <- TRAIN[, -which(NZV[1:nrow(NZV),]$nzv == TRUE)]
TEST <- TEST[, -which(NZV[1:nrow(NZV),]$nzv == TRUE)]
TRAIN$LABEL <- LABEL



# define xgb.train parameters
PARAM <- list(
  # General Parameters
  booster            = "gbtree",          # default
  silent             = 0,                 # default
  # Booster Parameters
  eta                = 0.05,              # default = 0.30
  gamma              = 0,                 # default
  max_depth          = 5,                 # default = 6
  min_child_weight   = 1,                 # default
  subsample          = 0.70,              # default = 1
  colsample_bytree   = 0.95,              # default = 1
  num_parallel_tree  = 1,                 # default
  lambda             = 0,                 # default
  lambda_bias        = 0,                 # default
  alpha              = 0,                 # default
  # Task Parameters
  objective          = "multi:softmax",   # default = "reg:linear"
  num_class          = 10,                # default = 0
  base_score         = 0.5,               # default
  eval_metric        = "merror"           # default = "rmes"
)


# convert TRAIN dataframe into a design matrix
TRAIN_SMM <- sparse.model.matrix(LABEL ~ ., data = TRAIN)
TRAIN_XGB <- xgb.DMatrix(data = TRAIN_SMM, label = LABEL)


#start time
ptm <- proc.time()

# train xgb model
MODEL <- xgb.train(params      = PARAM, 
                   data        = TRAIN_XGB, 
                   nrounds     = 50, # change this to 400
                   verbose     = 2,
                   watchlist   = list(TRAIN_SMM = TRAIN_XGB)
)

#end time
time_xgb_NN <- proc.time() - ptm	
time_xgb_NN <- time_xgb_NN[[3]]

time_xgb_NN <- 251.52




TEST$LABEL <- 0

# use the trained xgb model ("MODEL") on the test data ("TEST") to predict the response variable ("LABEL")
TEST_SMM <- sparse.model.matrix(LABEL ~ ., data = TEST)
PRED <- predict(MODEL, TEST_SMM)

