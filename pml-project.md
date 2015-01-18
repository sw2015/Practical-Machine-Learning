# Project for Practical Machine Learning

## Background

Six people perform barbell lifts. The lifts are classified in 5 different ways. 

1. exactly according to the specification (Class A)
2. throwing the elbows to the front (Class B)
3. lifting the dumbbell only halfway (Class C)
4. lowering the dumbbell only halfway (Class D)
5. throwing the hips to the front (Class E)

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 


## Data Preparation

Download the data to the local computer: 

training set: pml-training.csv

testing set: pml-testing.csv

## Load Libraries

```

library(ggplot2)
library(caret)
library(randomForest)

```

## Clean Data

Load to R by replacing empty entries with NA’s

```
training <- read.csv("~/Desktop/machine learning/pml-training.csv", na.strings = c("", "NA"))

testing <- read.csv("~/Desktop/machine learning/pml-testing.csv", na.strings = c("", "NA"))

```

Remove empty entries and NA’s

```

training <- training[, colSums(is.na(training)) == 0]

testing <- testing[, colSums(is.na(testing)) == 0]

```

Remove the first seven columns in the training set and testing set which are unrelated to the activities:

``` 

training <- training[-c(1:7)]

testing <- testing[-c(1:7)]

```

There are a total of 53 columns after cleaning


```

colnames(training)

 [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
 [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
 [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
[10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
[13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
[16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
[19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
[22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
[25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
[28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
[31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
[34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
[37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
[40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
[43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
[46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
[49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
[52] "magnet_forearm_z"     "classe" 

```

## Data Partition

```

inTrain <- createDataPartition(training$classe, p = 0.7, list = FALSE)

train <- training[inTrain, ]

test <-  training[-inTrain, ]

```

## Model Build

Use random forest algorithm

```
modelFit <- train(classe ~ ., data = train, method = "rf", trControl = trainControl(method = "cv", number = 5))

```

Here I am using k-fold cross-validation to sample the data with 5 folds instead of bootstrapping


## Data Prediction

Use the test data that we sampled from training set:

```
pred <- predict(modelFit, test)

```

## Out of Sample Error

Use Confusion Matrix to compute out of sample error:

1 - 0.9932 = 0.0068 = 0.68%


```

confusionMatrix(test$classe, pred)

Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1673    0    1    0    0
         B    9 1128    2    0    0
         C    0    8 1015    3    0
         D    0    0   11  952    1
         E    0    1    3    1 1077

Overall Statistics
                                          
               Accuracy : 0.9932          
                 95% CI : (0.9908, 0.9951)
    No Information Rate : 0.2858          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9914          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9946   0.9921   0.9835   0.9958   0.9991
Specificity            0.9998   0.9977   0.9977   0.9976   0.9990
Pos Pred Value         0.9994   0.9903   0.9893   0.9876   0.9954
Neg Pred Value         0.9979   0.9981   0.9965   0.9992   0.9998
Prevalence             0.2858   0.1932   0.1754   0.1624   0.1832
Detection Rate         0.2843   0.1917   0.1725   0.1618   0.1830
Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
Balanced Accuracy      0.9972   0.9949   0.9906   0.9967   0.9990

```

Note that accuracy is 0.9911 in the training data with 5-fold cross-validation as mtry = 27 is used:

```

modelFit
Random Forest 

13737 samples
   52 predictor
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Cross-Validated (5 fold) 

Summary of sample sizes: 10990, 10991, 10989, 10989, 10989 

Resampling results across tuning parameters:

  mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
   2    0.9907548  0.9883033  0.002339049  0.002960321
  27    0.9911187  0.9887649  0.000800172  0.001011936
  52    0.9857319  0.9819513  0.004952988  0.006264063

Accuracy was used to select the optimal model using  the largest value.
The final value used for the model was mtry = 27.

```




## Test Data

```

predict(modelFit, newdata = testing)

[1] B A B A A E D B A A B C B A E E A B B B
Levels: A B C D E

```
