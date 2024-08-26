
# For Nourishment and Interest
# Post Deadline for Exercise Submission
# Fit a Neural Network to the data

# Results
### Implementing a Neural Network on this data set to predict GDP, is not worth it
### The trade-off in complexity and efficiency is not worth it.
### A simple linear regression model should be used.

### Because of the simplicity of the relationship between GDP and the explanatory variables, 
### linear regression is preferred

library(tidyverse)
library(tictoc)
library(keras)
library(reticulate)
library(tensorflow)

library(ggplot2)
library(dplyr)
library(imputeTS)

# Load the Dataset

data = read.csv("data/data.csv")

# See that each variable has a NA value
# See that the max for each variable is two degrees greater than even   the 3rd Quartile Statistics

summary(data)

# Impute Missing Values with Linear Weighted Moving Average

newData = na_ma(data, k = 2, weighting = "linear")

# Clean Data for added two zeroes on single values in each variable

maxGDP = max(newData$GDP)
maxGFCF = max(newData$GFCF)
maxUNEM = max(newData$UNEM)
maxConsP = max(newData$ConsumerPrices)
maxGovExp = max(newData$GovExp)
maxHE = max(newData$HouseExp)

newData$GDP[newData$GDP >= maxGDP] = maxGDP/100
newData$GFCF[newData$GFCF >= maxGFCF] = maxGFCF/100
newData$UNEM[newData$UNEM >= maxUNEM] = maxUNEM/100
newData$ConsumerPrices[newData$ConsumerPrices >= maxConsP] = maxConsP/100
newData$GovExp[newData$GovExp >= maxGovExp] = maxGovExp/100
newData$HouseExp[newData$HouseExp >= maxHE] = maxHE/100

newData$Date = as.Date(newData$Date)

# Split Data into Training and Test Data

set.seed(2024)

mod_Data = newData[, -1]

mod_Data$id = 1:nrow(mod_Data)

# Split 75% Training Data and 25% Test Data
# Not ideal for such a small sample set

trainData = mod_Data %>% sample_frac(0.75)
testData  = anti_join(mod_Data, trainData, by = 'id')

trainData = trainData[, -7]
testData = testData[, -7]

Train_X = trainData[, -1]
Train_X = scale(Train_X)

Test_X = testData[, -1]
Test_X = scale(Test_X, center = attr(Train_X, "scaled:center"), 
               scale = attr(Train_X, "scaled:scale"))

Train_Y = scale(trainData[, 1])
Test_Y = scale(testData[, 1], center = attr(Train_Y, "scaled:center"), 
               scale = attr(Train_Y, "scaled:scale"))

dim = dim(Train_X)
nnMod = keras_model_sequential()

nnMod %>% 
  layer_dense(units = 32, 
              activation = "relu", 
              input_shape = dim[2]) %>% 
  layer_dense(units = 16, 
              activation = "relu") %>%
  layer_dense(units = 1, 
              activation = 'linear')

summary(nnMod)

nnMod %>% compile(
  loss = "mse",
  optimizer = optimizer_adam(learning_rate = 0.01),
  metrics = c("mse")
)

nnMod.History = nnMod %>% fit(
  as.matrix(Train_X), as.matrix(Train_Y), 
  epochs = 200, batch_size = 128,
  validation_split = 0.1,
  verbose = 1
)

predY = nnMod %>% predict(Test_X)

preds = data.frame(trueY = round(Test_Y, 4), predY = round(predY, 4))
preds

mse = mean((Test_Y - predY)^2)

### Implementing a Neural Network on this data set to predict GDP, is not worth it
### The trade-off in complexity and efficiency is not worth it.
### A simple linear regression model should be used.
