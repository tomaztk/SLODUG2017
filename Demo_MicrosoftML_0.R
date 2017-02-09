###
### SLO_DUG 09.02.2017
### Leaf Classification
### MicrosoftML - Demo
###

# NOTE: Make sure you connect to R Engine: 64Bit - [64-bit] C:\Program Files\Microsoft SQL Server\140\R_SERVER
# .libPaths()
# Tools -> Global Options
setwd("C:/DataTK/Kaggle/Leaf_Classification/")

##############################################################

## ---- echo = FALSE, message = FALSE--------------------------------------
#knitr::opts_chunk$set(collapse = TRUE, comment = "#>")
#options(tibble.print_min = 4L, tibble.print_max = 4L) 



library(MicrosoftML)
library(RevoScaleR)
library(lattice)


# Read the data into a data frame in memory
mortXdf <- file.path(rxGetOption("sampleDataDir"), "mortDefaultSmall")
mortDF <- rxDataStep(mortXdf)


# Create a logical TRUE/FALSE variable
mortDF$default <- mortDF$default == 1

# Divide the data into train and test data sets
set.seed(37)
isTest <- rnorm(nrow(mortDF)) > 0
mortTest <- mortDF[isTest,]
mortTrain <- mortDF[!isTest,]

#information on Data
rxGetInfo(mortTrain, getVarInfo = TRUE)
rxGetInfo(mortTest)



##################################
# Binary type labels
##################################

# Binary Formula

binaryFormula <- default ~ creditScore + houseAge + yearsEmploy + ccDebt + year

#-----------------------
# rxLogisticRegression
#-----------------------

logitModel <- rxLogisticRegression(binaryFormula, data = mortTrain) #add: showTrainingStats = TRUE
summary(logitModel)
logitScore <- rxPredict(logitModel, data = mortTest, extraVarsToWrite = "default")

#draw ROC Curve
rxRocCurve(actualVarName = "default", predVarNames = "Probability", data = logitScore)
# AUC = 0.90

#-----------------------
# rxFastTrees
#-----------------------

fastTreeModel <- rxFastTrees(binaryFormula, data = mortTrain, numTrees = 75, numLeaves = 10)
summary(fastTreeModel)
fastTreeScore <- rxPredict(fastTreeModel, data = mortTest,  extraVarsToWrite = "default")


ftRoc <- rxRoc(actualVarName = "default", predVarNames = "Probability", data = fastTreeScore)
rxAuc(ftRoc)
# [1] 0.9326229


#-----------------------
# rxFastForest
#-----------------------


rxFastForestModel <- rxFastForest(binaryFormula, data = mortTrain, numTrees = 75, numLeaves = 10)
summary(rxFastForestModel)
rxFastForestScore <- rxPredict(rxFastForestModel, data = mortTest,  extraVarsToWrite = "default")
#Roc
ffRoc <- rxRoc(actualVarName = "default", predVarNames = "Probability",data = rxFastForestScore)
rxAuc(ffRoc)
# [1] 0.7441262


# ------------------
# rxNeuralNet
# ------------------
rxNeuralNetModel <- rxNeuralNet(binaryFormula, data = mortTrain, numHiddenNodes = 10)
rxNeuralNetScore <- rxPredict(rxNeuralNetModel, data = mortTest,  extraVarsToWrite = "default")
rxRocCurve(actualVarName = "default", predVarNames = "Probability",
           data = rxNeuralNetScore, title = "ROC Curve for 'default' using rxNeuralNet")
# AUC 0.76


# ------------------
# rxFastLinear
# ------------------
rxFastLinearModel <- rxFastLinear(binaryFormula, data = mortTrain)
summary(rxFastLinearModel)
rxFastLinearScore <- rxPredict(rxFastLinearModel, data = mortTest,  extraVarsToWrite = "default")
rxRocCurve(actualVarName = "default", predVarNames = "Probability",    
           data = rxFastLinearScore, title = "ROC Curve for 'default' using rxFastLinear")
# AUC =  0.95



##################################
# Multi-class type labels
##################################



trainRows <- c(1:30, 51:80, 101:130)
testRows = !(1:150 %in% trainRows)
trainIris <- iris[trainRows,]
testIris <- iris[testRows,]
multiFormula <- Species ~Sepal.Length + Sepal.Width + Petal.Length + Petal.Width


#-----------------------
# rxLogisticRegression
#-----------------------

logitModel <- rxLogisticRegression(multiFormula, type = "multiClass", data = trainIris)
logitScore <- rxPredict(logitModel, data = testIris,  extraVarsToWrite = "Species")

#predicted Labels
rxCrossTabs(~Species:PredictedLabel, data = logitScore, removeZeroCounts = TRUE)


# ------------------
# rxNeuralNet
# ------------------

rxNeuralNetModel <- rxNeuralNet(
                                   multiFormula
                                  ,type = "multiClass"
                                  ,optimizer = sgd(learningRate = 0.2)
                                  ,data = trainIris
                                )

rxNeuralNetScore <- rxPredict(rxNeuralNetModel, data = testIris,  extraVarsToWrite = "Species")

#predicted Labels
rxCrossTabs(~Species:PredictedLabel, data = rxNeuralNetScore, removeZeroCounts = TRUE)



##################################
# Regression type label
##################################


# Sample Data
DF <- airquality[!is.na(airquality$Ozone), ]
DF$Ozone <- as.numeric(DF$Ozone)
randomSplit <- rnorm(nrow(DF))
trainAir <- DF[randomSplit >= 0,]
testAir <- DF[randomSplit < 0,]



# Regression type label formula
airFormula <- Ozone ~ Solar.R + Wind + Temp



# ------------------------
# rxFastTrees
# ------------------------

fastTreeModel <- rxFastTrees(airFormula, type = "regression",  data = trainAir)
fastTreeScore <- rxPredict(fastTreeModel, data = testAir,  extraVarsToWrite = "Ozone")
rxLinePlot(Score~Ozone, type = c("smooth", "p"), data = fastTreeScore,
           title = "rxFastTrees", lineColor = "red")


# ------------------------
# rxFastForest
# ------------------------

rxFastForestModel <- rxFastForest(airFormula, type = "regression",  data = trainAir)
rxFastForestScore <- rxPredict(rxFastForestModel, data = testAir,  extraVarsToWrite = "Ozone")
rxLinePlot(Score~Ozone, type = c("smooth", "p"), 
           data = rxFastForestScore, title = "rxFastForest", lineColor = "red")


# ------------------------
# rxNeuralNet
# ------------------------

rxNeuralNetModel <- rxNeuralNet(airFormula, type = "regression",  data = trainAir, numHiddenNodes = 8)
rxNeuralNetScore <- rxPredict(rxNeuralNetModel, data = testAir,  extraVarsToWrite = "Ozone")

rxLinePlot(Score~Ozone, type = c("smooth", "p"), data = rxNeuralNetScore,
           title = "rxNeuralNet", lineColor = "red")


# ------------------------
# rxFastLinear with l1Weight and l2Weight
# ------------------------

rxFastLinearModel <- rxFastLinear(airFormula, type = "regression",  data = trainAir, l2Weight = 0.01)
rxFastLinearScore <- rxPredict(rxFastLinearModel, data = testAir, extraVarsToWrite = "Ozone")
rxLinePlot(Score~Ozone, type = c("smooth", "p"), data = rxFastLinearScore,
           title = "rxFastLinear", lineColor = "red")


# ------------------
# rxOneClassSvm
# ------------------

# generate some random data
numRows <- 500
normalData <- data.frame( day = 1:numRows)
normalData$pageViews = runif(numRows, min = 10, max = 1000) 
normalData$clicks = runif(numRows, min = 0, max = 5)

testData <- data.frame(day = 1:numRows)
testData$pageViews = runif(numRows, min = 10, max = 1000) 
testData$clicks = runif(numRows, min = 0, max = 5)

outliers <-c(100, 200, 300, 400)
testData$outliers <- FALSE
testData$outliers[outliers] <- TRUE
testData$pageViews[outliers] <- 950 + runif(4, min = 0, max = 50)
testData$clicks[outliers] <- 5 + runif(4, min = 0, max = 1)

# model train and prediction
modelSvm <- rxOneClassSvm(formula = ~pageViews + clicks,data = normalData)
score1DF <- rxPredict(modelSvm, data = testData,extraVarsToWrite = c("outliers", "day"))


rxLinePlot(Score~day, type = c("p"), data = score1DF,
           title = "Scores from rxOneClassSvm",
           symbolColor = ifelse(score1DF$outliers, "red", "blue"))


# --------------------------
# Sentiment analysis
# --------------------------

trainReviews <- as.data.frame(matrix(c(
  "TRUE",  "This is great",
  "FALSE", "I hate it",
  "TRUE",  "Love it",
  "FALSE", "Do not like it",
  "TRUE",  "Really like it",
  "FALSE", "I hate it",
  "TRUE",  "I like it a lot",
  "FALSE", "I kind of hate it",
  "TRUE",  "I do like it",
  "FALSE", "I really hate it",
  "TRUE",  "It is very good",
  "FALSE", "I hate it a bunch",
  "TRUE",  "I love it a bunch",
  "FALSE", "I hate it",
  "TRUE",  "I like it very much",
  "FALSE", "I hate it very much.",
  "TRUE",  "I really do love it",
  "FALSE", "I really do hate it",
  "TRUE",  "Love it!",
  "FALSE", "Hate it!",
  "TRUE",  "I love it",
  "FALSE", "I hate it",
  "TRUE",  "I love it",
  "FALSE", "I hate it",
  "TRUE",  "I love it"),
  ncol = 2, byrow = TRUE, dimnames = list(NULL, c("like", "review"))),
  stringsAsFactors = FALSE)
trainReviews$like <- as.logical(trainReviews$like)

testReviews <- data.frame(review = c(
  "This is great",
  "I hate it",
  "Love it",
  "Really like it",
  "I hate it",
  "I like it a lot",
  "I love it",
  "I do like it",
  "I really hate it",
  "I love it"), stringsAsFactors = FALSE)


# -- Check the results of sentiment with different Transforms

# ------------------------------------
# categorical with rxLogisticRegression
# --------------------------------------

outModel1 <- rxLogisticRegression(like~catReview, data = trainReviews, 
                                  mlTransforms = list(categorical(vars = c(catReview = "review"))))
summary(outModel1)
scoreOutDF1 <- rxPredict(outModel1, data = testReviews, extraVarsToWrite = "review")

scoreOutDF1



# -----------------------------------------
# categoricalHash with rxLogisticRegression
# ------------------------------------------

outModel2 <- rxLogisticRegression(like~hashReview, data = trainReviews, 
                                  mlTransforms = list(
                                    categoricalHash(vars = c(hashReview = "review"), invertHash = -1,
                                                    hashBits = 8 )))

summary(outModel2)

scoreOutDF2 <- rxPredict(outModel2, data = testReviews, extraVarsToWrite = "review")
scoreOutDF2

# -------------------------------------------
# featurizeText with rxLogisticRegression
# -------------------------------------------

outModel5 <- rxLogisticRegression(like~reviewTran, data = trainReviews, 
                                  mlTransforms = list(featurizeText(vars = c(reviewTran = "review"),
                                                                    stopwordsRemover = stopwordsDefault(), 
                                                                    keepPunctuations = FALSE)))

summary(outModel5)

scoreOutDF5 <- rxPredict(outModel5, data = testReviews, extraVarsToWrite = "review")
scoreOutDF5

# -------------------------------
# categoricalHash with rxNeuralNet
# --------------------------------

outModel6 <- rxNeuralNet(like ~ hashReview
                         ,data = trainReviews
                         ,optimizer = sgd(learningRate = 0.1)
                         ,mlTransforms = list(
                                categoricalHash(vars = c(hashReview = "review"), invertHash = 1, hashBits = 8),
                                selectFeatures("hashReview", mode = minCount())))

scoreOutDF6 <- rxPredict(outModel6, data = testReviews, extraVarsToWrite = "review")
scoreOutDF6











