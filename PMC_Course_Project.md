---
title: "Classification Learning and Recognition of Weight Lifting Exercises"
output: html_document
---
### Practical Machine Learning Course Project 
#### Synopsis
This paper applied practical machine learning techniques to data from *Qualitative Activity Recognition of Weight Lifting Exercise* by Velloso, et al. (hereafter, the Velloso Paper). The Velloso Paper specified five different classifications of "Unilateral Dumbbell Biceps Curl" performed by participants (one set of 10 repetitions) and those classifications served as the response variable. Two datasets were provided for this project from the above mentioned study: testing and training. [Citation Link for Data](http://groupware.les.inf.puc-rio.br/har). The training dataset was initially reviewed and explored to determine the necessary data processing for model building. With the processed data,
*out-of-sample* error rate was estimated and learning technique was applied. Finally, the trained model was applied to the test dataset.

#### Initial Data Review
The training dataset, in its unaltered form, had 19,622 observations with 160 variables. The first 7 variables were instance specific (for example, user names) and such variables were prone to *overfitting*; hence, those variables were set aside.

In addition, there were 100 variables with **NA** values and even instances of **#DIV/0!** errors (often seen on spreadsheets). The decision to eliminate variables with instances of **#DIV/0!** was an easy one; however, variables with **NA** instances had to be reviewed further. In the Velloso Paper Section 5.1, the authors stated that the measurements were on the Euler angles (roll, pitch and yaw) and that 96 variables were generated as calculated values (descriptive statistical measures) for Euler angles of each of the four sensors. Hence, the 96 variables were not pure measurements but summarized values of the 
measurements. Under this scenario, it would not be prudent to retain those variables or fill-in those variables as such attempts would adulterate the training dataset.

The initial data review yielded a list of variables for elimination for the first round of analysis.

#### Data Processing
The training dataset was loaded into the R environment excluding the columns (or variables) with **NA** and **#DIV/0!** instances; and the dataset was subset into predictors and response datasets.

```r
require(RCurl)
```

```
## Loading required package: RCurl
## Loading required package: bitops
```

```r
url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
# Need to remove columns with error messages and NA
delimDt <- read.delim(textConnection(getURL(url)),
                      header=TRUE,
                      sep=",",
                      na.strings=c("#DIV/0!","NA"))
# Additional step to ensure that feature columns do nat have NAs
dtN <- delimDt[ , apply(delimDt, 2, function(x) !any(is.na(x)))]
# Eliminate instance specific variables
training <- dtN[,8:60]
# Subsets
trainX <- training[,1:52]
trainY <- as.data.frame(training[,53])
```
The necessary datasets were generated for further analysis.

#### Model Building

##### Feature Selection
With 52 predictors and 5 distinct classes, data visualization of the data was problematic (mostly due to lack of knowledge of data visualization techniques for high dimensional data). Hence, it was decided that gratuitous use of plots would be minimized. 

Other than the Velloso Paper, there was lack of domain knowledge to have meaningful feature (or predictor) selection that balances the number of features vs. summarization. Because of the lack of domain knowledge, it became apparent that predictor selection had to be automated via the PCA and the training model itself.

First, a simple contingency table was created for the *class* (named classe in the dataset) variable.

```r
classeTab <- table(training$classe)
round(classeTab/length(training$classe),2) # Each class as a percentage of total 
```

```
## 
##    A    B    C    D    E 
## 0.28 0.19 0.17 0.16 0.18
```
In the training dataset, *class* A was the modal class; however, each *class* had relatively similar proportion of the total. 

The main issue at hand was the 52 predictor variables that was distilled from 159 variables in the original dataset. The authors of the Velloso Paper indicated that the measurements were taken from four sensors in each of the Euler angles; hence, it was conceivable that some of the predictor variables might be correlated and would not provide incremental predictive value. Principal Component Analysis (PCA) was performed to see if some of the variables can be eliminated without loss of predictive power of the dataset as a whole.


```r
initPCA <- prcomp(trainX, scale.=TRUE)
# Determine the proporation of the variance explained by each principal component
varPCA <- ((initPCA$sdev)^2)
pvePCA <- varPCA/sum(varPCA)
cumSum <- cumsum(pvePCA); round(quantile(cumSum, c(.5,.75,.95)),2)
```

```
##  50%  75%  95% 
## 0.96 0.99 1.00
```
The results showed that with 50% of the principal components (approximately 25 variables) explain more than 95% of the total variance in the training dataset.

Based on the results above, features (or predictors) were preprocessed using the PCA in the caret package.

```r
# Load required packages
require(caret); require(randomForest)
```

```r
# Set seed for reproducibility
set.seed(1234)
# Generate PCA based features
preProc <- preProcess(trainX, method="pca")
trainPC <- predict(preProc,trainX)
```
##### Model Selection
Even though PCA reduced the number of features to a more manageable number, it was unclear on how to determine a mix of features that can predict the outcome without trying different combinations of
features. 

In Chapter 8 of *Introduction to Statistical Learning* (ISLR) by James, Witten, Hastie and Tibshirani, the authors stated that **Random Forest** (RF) provide additional benefits of *bagging* by decorrelating the trees. In RF, each time a split in a tree is considered, a random sample of *m* features are chosen from the full list of features and a fresh sample of *m* features are chosen at each split. (p. 320)

This description of the model seemed to address the concern of picking the mix of features. In addition, RF also retains the "majority vote" concept in *bagging*. However, even with all the benefits of the RF model, the concern for overfitting remained.

##### Errors and Cross-Validation
The out-of-sample error was of great concern. In *ISLR*, the authors mentioned that RF will not overfit if the number of trees are sufficiently large. (p. 321) Further, the authors stated that *k-fold cross-validation* would help to select the number of trees. The *randomForest* package help file noted that the default number of trees is 500 so that would be the starting point.

The decision was to use 10-fold cross-validation. The authors of *ISLR* mentioned that k =5 or k =10 is typical (p. 181). Since overfitting--thus, out-of-sample-error--was of great concern, k = 10 was chosen. Although larger k may yield greater variance than that of lower k, the possibility of lower bias 
associated with larger k was deemed more desirable.


```r
modFit10 <- train(training$classe ~ ., method="rf", data=trainPC,
                trControl=trainControl(method="cv", number=10))
confusionMatrix(modFit10)
```

```
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are percentages of table totals)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.3  0.3  0.0  0.0  0.0
##          B  0.0 18.9  0.2  0.0  0.0
##          C  0.1  0.1 17.1  0.6  0.1
##          D  0.0  0.0  0.1 15.8  0.1
##          E  0.0  0.0  0.0  0.0 18.2
```

```r
round(classeTab/length(training$classe),2)
```

```
## 
##    A    B    C    D    E 
## 0.28 0.19 0.17 0.16 0.18
```
From the confusion matrix, the error rates appeared to be less than 5% as accuracy rates correspond to the contingency table. In order to assess out-of-sample error rate, individual cross-validation accuracy was reviewed.

```r
modFit10$resample
```

```
##     Accuracy     Kappa Resample
## 1  0.9811513 0.9761625   Fold02
## 2  0.9806221 0.9754766   Fold01
## 3  0.9826707 0.9780744   Fold03
## 4  0.9806320 0.9755005   Fold06
## 5  0.9811417 0.9761444   Fold05
## 6  0.9811609 0.9761560   Fold04
## 7  0.9831804 0.9787200   Fold07
## 8  0.9821520 0.9774216   Fold10
## 9  0.9867482 0.9832329   Fold09
## 10 0.9836984 0.9793748   Fold08
```
From the cross-validation accuracy measures, the error rates appear to be less than 2%. However, as much as cross-validation technique is a sound approach, the actual error rate is expected to be greater than
2%. 

#### Conclusion
There was a great care to remove features that were instance specific like user names. In addition, data was cleaned to eliminate **NA** and **##DIV0!** from the dataset. Since there was a lack of domain knowledge, model based dimension reduction and feature selection were used (PCA). Further, the model employed also sampled features at each tree node split to distill the effective features for prediction. However, such approach was also prone to overfitting. 

The results showed less than 2% as the estimate of out-of-sample error. The test dataset was used to assess the model and the test error rate was 5% (19 correct out of 20 test cases).
