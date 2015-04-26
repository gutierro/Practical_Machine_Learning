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

subset <- cfs(classe~., pml.training)
f <- as.simple.formula(subset, "classe")

set.seed(33833) 
inTrain <- createDataPartition(pml.training$classe, p=3/4, list=FALSE)
trainData <- pml.training[inTrain, ]
testData <- pml.training[-inTrain, ]

controlRf <- trainControl(method="cv", 10)

modelRf <- train(f, data=trainData, method="rf", trControl=controlRf, ntree=50)
#modelRf <- randomForest(f, data=pml.training, importance = FALSE,ntree = 50,mtry=length(subset))

modelRf

predictRf <- predict(modelRf, testData)
cm<-confusionMatrix(testData$classe, predictRf)

accuracy <- as.numeric(cm$overall[1])
accuracy

oose <- 1 - accuracy
oose

result <- predict(modelRf, pml.testing)
result

