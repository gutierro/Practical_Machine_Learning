# Project : Practical Machine Learning


###Summary 

This project is based on the paper "Qualitative Activity Recognition of Weight Lifting Exercises" [1] The goal of this project is to predict the manner in which the user did the exercise.
I have used the Correlation Feature Selection (CFS) to identify the most relevant features, using a Random Forest approach for the predictor with 50 trees. The classifier was tested with 10-fold  cross-validation.


###Exploratory data analyses 
To improve the performance I have reduced the dimension of the data sets, removing the irrelevant features.
Exploring the testing data set, it can be observed several features that don't have a value (NA) and therefor they won't be relevant for the prediction. I reduce the original data sets to new ones with only the features with the sensors data and excluding the others, but including the class.


```r
library(caret)
library(rpart)
library(randomForest)
library(FSelector)

pml.training<-read.csv("pml-training.csv")
pml.testing<-read.csv("pml-testing.csv")

colNames<-names(pml.testing[,colSums(is.na(pml.testing)) != nrow(pml.testing)])
colNames<-colNames[colNames!="problem_id"]

pml.testing<-pml.testing[,c(colNames,"problem_id")]
pml.testing<-pml.testing[,-(1:7)]

pml.training<-pml.training[,c(colNames,"classe")]
pml.training<-pml.training[,-(1:7)]
```

###Relevant features selection
To improve the performance, I use the Correlation Feature Selection (CFS) algorithm to select the most relevant features. This algorithm evaluates subsets of features on the basis that the subsets contain features  that are highly correlated with the classification and uncorrelated to each other.
The cf function in R is configured to use a "Best First" strategy based on backtracking. (cf help for more information)


```r
set.seed(33833) 
subset <- cfs(classe~., pml.training)
f <- as.simple.formula(subset, "classe")
```

###Data Slicing
I split the data 75% for the training and 25% what a split commonly use and perform quite well for data sets of this size.


```r
inTrain <- createDataPartition(pml.training$classe, p=.75, list=FALSE)
trainData <- pml.training[inTrain, ]
testData <- pml.training[-inTrain, ]
```


###Predictive Model fitting
I use the Random Forest algorith for because is quiet accurate and it works very well with the noice.[1]  
For performance reasons I select the number of trees to 50. Increasing the number only improves slightly the accuracy but in contrast the performance gets highly deteriorated.


```r
controlRf <- trainControl(method="cv", 10)
modelRf <- train(f, data=trainData, method="rf", trControl=controlRf, ntree=50)
predictRf <- predict(modelRf, testData)
```

***There random Forest Package in R provides the random Forest function that offers a much better performance than the "train" function with "rf" and the packages has in addition other functions, specific for plotting the random Forest object. The function uses "Leave one out" type of cross validation error and it is combined with "Bagging" to improve the performance. The output is an object with that contains the model fit and it already calculates the confusion Matrix  and the "Out of bag" error. For this project I use the function "train" with "rf"  taught in the class, but add an example of randomForest as a reference.


```r
#modelRf2 <- randomForest(f, data=pml.training, importance = FALSE,ntree = 50,mtry=length(subset))
```



###Residual values analysis
We new that the random Forest model is quite accurate in this case , the accuracy is   98.23% and the stimated out-of-sample error 1.77%. Increasing the number of trees would improve the accuracy at a cost of a big downgrade of the performance. Therefore tuning the number of trees it would not bring any benefit.



```r
cm<-confusionMatrix(testData$classe, predictRf)
accuracy <- as.numeric(cm$overall[1])
oose <- 1 - accuracy
cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1374    5    8    8    0
##          B   10  917   16    1    5
##          C    1    9  837    4    4
##          D    4    0    2  798    0
##          E    0    6    1    3  891
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9823          
##                  95% CI : (0.9782, 0.9858)
##     No Information Rate : 0.2832          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9776          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9892   0.9787   0.9688   0.9803   0.9900
## Specificity            0.9940   0.9919   0.9955   0.9985   0.9975
## Pos Pred Value         0.9849   0.9663   0.9789   0.9925   0.9889
## Neg Pred Value         0.9957   0.9949   0.9933   0.9961   0.9978
## Prevalence             0.2832   0.1911   0.1762   0.1660   0.1835
## Detection Rate         0.2802   0.1870   0.1707   0.1627   0.1817
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9916   0.9853   0.9821   0.9894   0.9938
```

```r
accuracy
```

```
## [1] 0.9822594
```

```r
oose
```

```
## [1] 0.01774062
```


##Results
Applying the model to the testing data of this project's problem.

```r
result <- predict(modelRf, pml.testing)
result
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


##References
[1] Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

