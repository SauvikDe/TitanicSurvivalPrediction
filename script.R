#Author: Sauvik De
#Date: Dec. 25th, 2016
#Purpose: Kaggle competition
#Analysis: Predicting survival on Titanic disaster
#https://www.kaggle.com/c/titanic
#*************************************************

#remove all objects from current session
rm(list = ls())

#attach required libraries
library(ggplot2)
library(dplyr)
library(missForest)
library(mice)
library(randomForest)
library(reshape2)
library(gridExtra)

#set working directory
setwd("G:\\Kaggle\\Titanic")

#read in training and test datasets
train <- read.csv(file = "train.csv", header = TRUE, stringsAsFactors = FALSE)
test  <- read.csv(file = "test.csv",  header = TRUE, stringsAsFactors = FALSE)

#check how data structure is
str(train)
str(test)

#combine training and test datasets to form an overall dataset
test$Survived <- NA
full <- merge(x = train, y = test, all = TRUE)

#convert a few 'useful' categorical variables into class 'factor'
catVars <- c("Pclass", "Sex", "Embarked")
full[catVars] <- lapply(full[catVars], factor)

#get count of NA values; get a feel about data availability
colSums(is.na(full)) + colSums(full == "", na.rm = TRUE)

#feature engineering and some exploratory analysis
#=================================================
#Plot survival proportions by sex - females more likely survived than men
ggplot(data = train, mapping = aes(x = Sex, fill = factor(Survived))) + geom_bar(position = 'fill') + scale_fill_discrete(name = "Survived") + ylab("Proportion")

#Survival proportions by sex - in number
prop.table( table(train$Sex, train$Survived), 1 )

#Let's extract titles ..
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

#Look at the frequency distribution of passengers' titles
sort(table(full$Title))

#Substitute well-known english titles for those "unfamiliar" titles
full$Title[(full$Title == "Mme"  | full$Title == "Lady")] <- "Mrs"
full$Title[(full$Title == "Mlle" | full$Title == "Ms")]   <- "Miss"
full$Title[full$Title == "Sir"] <- "Mr"
full$Title[full$Title != "Mr" & full$Title != "Mrs" & full$Title != "Master" & full$Title != "Miss"] <- "OtherTitle"

#Look at the revised frequency distribution of passengers' titles
sort(table(full$Title))

full$Title <- factor(full$Title)

#Look at counts of titles by sex - to check for data-integrity - result looks reasonable
table(full$Sex, full$Title)

#Let's look at age distribution for survived/not survived passengers by title
#Generally, older people from each title category seem to have survived
ggplot(filter(full,!is.na(Survived)), aes(x = Title, y = Age, fill = factor(Survived))) + geom_boxplot() + scale_fill_discrete(name = "Survived") + xlab("Title")

#Also, let's look at the survival counts by title
#Except category "Mr" and "OtherTitle", there are more survivals than not
ggplot(filter(full,!is.na(Survived)), aes(x = Title, fill = factor(Survived))) + geom_bar(position = 'dodge') + scale_fill_discrete(name = "Survived") + xlab("Title")

#Get family size which may be a potential predictor for survival prediction exercise
full$FamilySize <- full$SibSp + full$Parch + 1

#Look at the count of survivals by family-size
#It looks like singleton and more than 4 family-size passengers were out of luck
ggplot(filter(full,!is.na(Survived)), aes(x = FamilySize, fill = factor(Survived))) + geom_bar(position = 'dodge', stat = 'count')

#Extract surnames for each passenger
full$Surname <- gsub('(,.*)', '', full$Name)

#Let's combine surname and family size to make a single variable
full$FamilyID <- paste(full$Surname, "_", full$FamilySize, sep = "")

full$Surname <- factor(full$Surname)
full$FamilyID <- factor(full$FamilyID)

#Count of families and max family-size
cat(paste("The data contains about ", length(unique(full$FamilyID)), " families, of which the maximum family-size was ", max(table(full$FamilyID)), sep = ""))

#Look at if survival vary across passenger class
#Notably, first class passengers had more survival; while third class passengers were out of luck!
ggplot(filter(full,!is.na(Survived)), aes(x = Pclass, fill = factor(Survived))) + geom_bar(position = 'dodge', stat = 'count') + scale_fill_discrete(name = "Survived") + xlab("Passenger Class")

#And, how does it vary by sex?
#Females, like before, seem to have survived more than males;
#Proportion of males survived also seems to have decreased steadily from first class through third class
ggplot(filter(full,!is.na(Survived)), aes(x = paste(Pclass, Sex, sep = "_"), fill = factor(Survived))) + geom_bar(position = 'fill', stat = 'count') + scale_fill_discrete(name = "Survived") + xlab("Passenger Class_Sex")

#missing value imputation
#========================
#NA count tells there are very few missing values in "Fare" and "Embarked"
#Let's try estimate them

#Passenger with missing 'Fare' value was travelling in third class (Pclass = 3), embarked from Southampton (Embarked = S)
filter(full, is.na(Fare))

#Compute summary statistics for fare group by passenger class = 3 and embark location = "S"
summary(filter(full, Embarked != "" & Pclass == 3 & Embarked == "S")$Fare)

#and, mode
names(which.max(table(filter(full, Embarked != "" & Pclass == 3 & Embarked == "S")$Fare)))

#look at histogram for distribution of fare for third class and embarkation from (S)outhampton
par(mfrow = c(1,2))
#this is pretty much a right-skewed distribution
ggplot(filter(full, Embarked != "" & Pclass == 3 & Embarked == "S"), aes(x = Fare)) + geom_histogram(binwidth = 4)

#density plot showing peak at the median
ggplot(full[full$Pclass == 3 & full$Embarked == 'S', ], aes(x = Fare)) + geom_density(fill = '#ba1639', alpha = 0.4) + geom_vline(aes(xintercept=median(Fare, na.rm=T)), colour='red', linetype='dashed', lwd=1) + scale_x_continuous()

#median = mode = 8.05 - we can very well use this as an estimate of the missing fare value
full[is.na(full$Fare),]$Fare <- median(filter(full, Embarked != "" & Pclass == 3 & Embarked == "S")$Fare, na.rm = TRUE)
#######

#Passenger with missing 'Embarked' value was travelling in first class (Pclass = 1), ticker fare was 80 pound (Fare = 80)
filter(full, Embarked == "")

#Compute summary statistics for fare group by passenger class and embark location
filter(full, Embarked != "") %>% group_by(Pclass, Embarked) %>% summarise(FareMedian = median(Fare, na.rm = TRUE), FareAvg = mean(Fare, na.rm = TRUE), FareMax = max(Fare, na.rm = TRUE), FareMin = min(Fare, na.rm = TRUE), Count = sum(!is.na(Fare)))

summary(filter(full, Embarked != "" & Pclass == 1 & Embarked == "S")$Fare)

#scatter plot
#ggplot(filter(full, Embarked != ""), aes(x = factor(PassengerId), y = Fare)) + geom_point(aes(shape = factor(Pclass), colour = factor(Embarked))) + scale_colour_discrete() + labs(shape = "Passenger class", colour = "Embarked") + theme(axis.title.x = element_blank(),axis.text.x = element_blank(),axis.ticks.x = element_blank())
#I removed a few seemingly bigger(outlier) fare values to have a better view of the plot - but this is still not conclusive enough
ggplot(filter(full, Embarked != "" & Fare < 500), aes(x = PassengerId, y = Fare)) + geom_point(aes(shape = Pclass, colour = Embarked)) + scale_colour_discrete() + labs(shape = "Passenger class", colour = "Embarked") + xlab("Passenger") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

#I try to look how the scatter looks around 80 - S and C seem to be "almost" equally dominant
ggplot(filter(full, Embarked != "" & Fare > 75 & Fare < 85), aes(x = PassengerId, y = Fare)) + geom_point(aes(shape = Pclass, colour = Embarked)) + scale_colour_discrete() + labs(shape = "Passenger class", colour = "Embarked") + xlab("Passenger") + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

#Looking closely, more S appears in the neighbourhood of Fare = 80 line; I will go for S and classify Embarked = S
full[full$Embarked == "",]$Embarked <- "S"
#######

#we can't do much with "Cabin" variable with lot of missing values;
#######

#"Age" variable has quite a few missing values too but it should be manageable and can be expected to be a significant variable in predicting survival
dim(filter(full, is.na(full$Age)))

#need to use some missing imputation alogorithm to fill in missing ages
#create a dataframe with relevant variables to predict age variable
inputDF <- full[c("Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title")]

#ensure we have only missing data from Age
colSums(is.na(inputDF)) + colSums(inputDF == "", na.rm = TRUE)

#I use MICE as it seems to do slightly better than missForest in this example by looking at the histogram; alternatively we could use decision tree
#running missForest ..
#using missForest function to impute the 'Age' variable; alternatively, we could use MICE/decision tree
set.seed(1010)
mfObj <- missForest(xmis = inputDF, maxiter = 10, ntree = 100)
#out-of-bag NRMSE - looks reasonable here
mfObj$OOBerror
#check all missing Age are imputed
sum(is.na(mfObj$ximp$Age))

#running MICE ..
#ran a single imputation using MICE
set.seed(1010)
miObj <- mice(inputDF, m = 1, method = "rf")
#Ensure all missing Ages are imputed
sum(is.na(complete(miObj)$Age))

#compare original data, imputed dataset using missForest and using MICE
par(mfrow = c(1,3))
hist(inputDF$Age, freq = FALSE, ylim = c(0,0.05), xlab = "Age", main = "Original_data") #original
hist(mfObj$ximp$Age, freq = FALSE, ylim = c(0,0.05), xlab = "Age", main = "missForest") #missForest
hist(complete(miObj)$Age, freq = FALSE, ylim = c(0,0.05), xlab = "Age", main = "MICE")  #MICE

#choose MICE and use the imputed values in the dataset
full$Age <- complete(miObj)$Age
#######

#get count of NA values; great - we have imputed all the variables except Cabin which we are going to ignore for this analysis
colSums(is.na(full)) + colSums(full == "", na.rm = TRUE)

#building model
#==============
#Okay, we are now all set to build our model to predict survival
#I will build a model using Breiman's Random Forest as this works better than trees for it helps circumvent overfitting
#and also it works great in terms of accuracy determined by Gini coefficient

#first segregate the training and test datasets
trainNew <- full[ (full$PassengerId %in% train$PassengerId), ]
testNew  <- full[ !(full$PassengerId %in% train$PassengerId), ]

#build model on training dataset
set.seed(1010)
rfMdl <- randomForest(formula = factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title, data = trainNew, ntree = 500, mtry = 3, importance = TRUE)

#check out how error rate behaves as we grow more and more trees
#and, notably the model is better off predicting non-survival cases than survivals
ggplot(melt(cbind.data.frame(as.data.frame(rfMdl$err.rate), Tree_count = c(1:nrow(rfMdl$err.rate))), id.vars = "Tree_count", variable.name = "Sample", value.name = "Error_Rate"), aes(x = Tree_count, y = Error_Rate, group = Sample, colour = Sample)) + geom_line(aes(y = Error_Rate))

#the confusion matrix to see the error rate numerically at ntree = 500
rfMdl$confusion

#also,look at the importance of each predictor
rfMdl$importance

impGini <- data.frame(impGini = rfMdl$importance[order(rfMdl$importance[,"MeanDecreaseGini"], decreasing = TRUE),"MeanDecreaseGini"])
impAccuracy <- data.frame(impAccu = rfMdl$importance[order(rfMdl$importance[,"MeanDecreaseAccuracy"], decreasing = TRUE),"MeanDecreaseAccuracy"])

impGini <- data.frame(variable = names(rfMdl$importance[,"MeanDecreaseGini"]), meanDecreaseGini = rfMdl$importance[,"MeanDecreaseGini"], row.names = NULL)
impGini$variable <- factor(impGini$variable, levels = impGini[order(impGini$meanDecreaseGini, decreasing = TRUE),"variable"])
g1 <- ggplot(impGini, aes(x = variable, y = meanDecreaseGini)) + geom_point()

impAccuracy <- data.frame(variable = names(rfMdl$importance[,"MeanDecreaseAccuracy"]), meanDecreaseAccuracy = rfMdl$importance[,"MeanDecreaseAccuracy"], row.names = NULL)
impAccuracy$variable <- factor(impAccuracy$variable, levels = impAccuracy[order(impAccuracy$meanDecreaseAccuracy, decreasing = TRUE),"variable"])
g2 <- ggplot(impAccuracy, aes(x = variable, y = meanDecreaseAccuracy)) + geom_point()

grid.arrange(g1, g2, nrow = 1, ncol = 2)

#predicting survival
#===================
testNew$Survived <- predict(rfMdl, newdata = testNew)

#write output to a csv file
#==========================
write.csv(cbind.data.frame(PassengerId = testNew$PassengerId, Survived = testNew$Survived), "modelSurvivalPrediction.csv", row.names = FALSE)
