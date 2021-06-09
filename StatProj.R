## MNIST Dataset Classification using Deep Learning:

#1 - Loading required libraries:
devtools::install_github("rstudio/keras")  #skip if already installed
library(keras)
install_keras()
library(tensorflow)
install_tensorflow(version = "nightly") #skip if already installed
library(keras)

#2 - Loading the Dataset:
mnist <- dataset_mnist()
x_train <- mnist$train$x
str(x_train)
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

dim(x_train)

#3 - Preprocessing:
  # Our x data is in a 3-d array (images,width,height)
  # Converting the 3-d arrays into matrices by reshaping width and height into a single dimension for training a MLP
  # reshape: 28*28 = 784 (28x28 images are flattened into length 784 vectors)

#Reshaping array to CNN 3D Input_Shape format by adding channel i.e, (width,height,channel):
X_train_CNN <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
X_test_CNN <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))
str(X_train_CNN)

str(x_train)

#Reshaping array to MLP Input Format:
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
str(x_train)
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# Converting the grayscale values from integers ranging between 0 to 255 into floating point values 
# ranging between 0 and 1 (Rescaling):
x_train <- x_train / 255
x_test <- x_test / 255

X_train_CNN <- X_train_CNN / 255
X_test_CNN <- X_test_CNN / 255

# y data contains labels from 0 to 9. For training we one-hot encode the vectors into binary class matrices using the 
# Keras to_categorical() function (hence we will use the categorical loss function):
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


#4 - Defining the Network Architecture:
#Using Multi-layer Perceptron:
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', 
              input_shape = c(784)) %>%    
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

# The input_shape argument to the first layer specifies 
# the shape of the input data (a length 784 numeric vector 
# representing a grayscale image). The final layer 
# outputs a length 10 numeric vector (probabilities for 
# each digit) using a softmax activation function.

summary(model)

# compiling the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# fitting the model in batches of 128 images and 30 epochs
history <- model %>% fit(
  x_train, y_train, 
  epochs = 20, batch_size = 128, 
  validation_split = 0.2
)

plot(history)

# evaluate accuracy
model %>% evaluate(x_test, y_test)

# prediction
model %>% predict_classes(x_test)


# MLP using regularization:
modelReg <- keras_model_sequential() 

modelReg %>% 
  layer_dense(units = 256, activation = 'relu', 
              input_shape = c(784),
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%    
  layer_dense(units = 128, activation = 'relu',
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(modelReg)

modelReg %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

historyReg <- modelReg %>% fit(
  x_train, y_train, 
  epochs = 20, batch_size = 128, 
  validation_split = 0.2
)


# evaluate accuracy
modelReg %>% evaluate(x_test, y_test)

# prediction
modelReg %>% predict_classes(x_test)

#Using CNN:
modelCNN <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", 
                input_shape = c(28, 28, 1)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = "softmax")

summary(modelCNN)  

modelCNN %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)


historyCNN <- modelCNN %>% 
  fit(
    X_train_CNN, y_train,
    epochs = 20,
    batch_size = 128, validation_split = 0.2
  )

# evaluate accuracy
modelCNN %>% evaluate(X_test_CNN, y_test)

# prediction
modelCNN %>% predict_classes(X_test_CNN)

#Comparing the 3 models:

library(tidyr)
library(tibble)
library(dplyr)
library(ggplot2)

compare_cx <- data.frame(
  MLP_train = history$metrics$loss,
  MLP_val = history$metrics$val_loss,
  L2_train = historyReg$metrics$loss,
  L2_val = historyReg$metrics$val_loss,
  CNN_train = historyCNN$metrics$loss,
  CNN_val = historyCNN$metrics$val_loss
) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)

ggplot(compare_cx, aes(x = rowname, y = value, color = type)) +
  geom_line() +
  xlab("epoch") +
  ylab("loss")
