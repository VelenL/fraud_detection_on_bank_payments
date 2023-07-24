library(ggplot2)
library("caret")
library(magrittr)
library(dplyr)
library("plyr")
#-----------------------------------
# Import data
#-----------------------------------
fr<-read.csv("fraud.csv", header=TRUE, stringsAsFactors = T)
#---------------------------------
# Data structure/cleaning
#---------------------------------
str(fr)
fr$step <- as.factor(fr$step)
fr$fraud <- as.factor(fr$fraud)
str(fr)

# Remove unnecessary columns
fr_sup=fr[,-c(2,5,6,7)]
fr_sup
str(fr_sup)
summary(fr_sup)

# Check for number of missing values
sum(is.na(fr_sup))
# no missing values

# Split into 2 groups
colnames(fr_sup)
df_fraud <- fr_sup[fr_sup$fraud == 1,] 
df_non_fraud <- fr_sup[fr_sup$fraud == 0,]

# Barplot
ggplot(fr_sup, aes(x=fraud, fill = fraud)) + geom_bar() +
  scale_fill_manual(values = c("green", "red"))+ggtitle("Count of Fraudulent Payments")

print(paste("Number of normal examples: ", nrow(df_non_fraud)))
print(paste("Number of fradulent examples: ", nrow(df_fraud)))

# Boxplot
ggplot(fr_sup, aes(x=category, y=amount)) +
  geom_boxplot() +
  ggtitle("Boxplot for the Amount spend in category") +
  ylim(0, 4000) +
  theme(axis.text.x = element_text(angle = 45, hjust = 0.5, vjust = 0.5))

# Histogram
ggplot() +
  geom_histogram(data = df_fraud, aes(x = amount), alpha = 0.5, fill = "red", bins=100) +
  geom_histogram(data = df_non_fraud, aes(x = amount), alpha = 0.5, fill = "blue", bins=100) +
  ggtitle("Histogram for fraudulent and nonfraudulent payments")+
  scale_x_continuous(limits = c(0, 1000)) +
  scale_y_continuous(limits = c(0, 4000))+
  scale_fill_manual(name="", values = c("red"="fraud", "blue"="nonfraud"))+
  theme(legend.position = "top")

# Percentage 
fr_sup$fraud <- as.numeric(fr_sup$fraud)
fr_sup$age <- as.numeric(fr_sup$age)

fr_sup%>% group_by(category) %>%summarize(mean_age = mean(age))

# ########
# df_fraud_mean <- df_fraud %>% group_by(category) %>% summarize(mean_amount = mean(amount))
# df_non_fraud_mean <- df_non_fraud %>% group_by(category) %>% summarize(mean_amount = mean(amount))
# df_fraud_percentage <- fr_sup %>% group_by(category) %>% summarize(fraud_percentage = mean(fraud)*100)
# 
# result <- rbind.fill(df_fraud_mean, df_non_fraud_mean[,1], df_fraud_percentage[,2])
# colnames(result) <- c("Fraudulent","Non-Fraudulent","Percent(%)")
# result <- result[order(result$`Non-Fraudulent`),]
# result
# Logistic regression

#-------------------------
# Model estimation
#-------------------------
fr_sup$fraud <- as.factor(fr_sup$fraud)
set.seed(1)
row.number <- sample(1:nrow(fr_sup), 0.8*nrow(fr_sup))
train=fr_sup[row.number,]
test=fr_sup[-row.number,]
dim(train)
dim(test)

mod.1 <- glm(fraud ~ step + age + gender + category + amount, data=train, binomial(link="logit"))
summary(mod.1)
mod.2 <- glm(fraud ~ age+ gender + category + amount, data=train, binomial(link="logit"))
summary(mod.2)


## We can continue with these predictors right now 

#----------------
# Model accuracy
#----------------

varImp(mod.2)

# amount,category'es_sportsandtoys', category'es_leisure' are the three most important values

prediction_test <- predict(mod.2, newdata = test, type = "response")

prop.table(table(test$fraud, prediction_test > 0.5))


# Almost 98.7% of the client who didn't have fraud have been correctly predicted
# but only 0.7% of the client who did have fraud have been correctly predicted

#---------------#
# Random Forest #
#---------------#
# use SMOTE 
# 
# remotes::install_github("cran/DMwR")
# 
# install.packages("devtools")
# library(devtools)
# install_github("R-imbalanced-learn/imbalanced-learn")
# 
# install.packages("imbalanced-learn")
# library(imbalanced-learn)
# install.packages("ROSE")
# install.packages("SMOTE")
# library(ROSE)
# library(SMOTE)
# set.seed(42)
# smote = SMOTE(fr_sup,ratio = "auto", random_state = 42)
# ?SMOTE
# X_res <- smote$fit_resample(X, y)
# y_res <- X_res[,ncol(X_res)]
# table(y_res)
# 
# install.packages('randomForest')
# library(randomForest)
# rf_clf <- randomForest(x = X_train, y = y_train, ntree = 100, maxnodes = 8,
#                        seed = 42, verbose = 1, classwt = "balanced")
# 
# y_pred <- predict(rf_clf, X_test)
# 
# print("Classification Report for Random Forest Classifier: \n")
# print(classification_report(y_test, y_pred))
# print("Confusion Matrix of Random Forest Classifier: \n")
# print(confusionMatrix(y_test,y_pred))
# plot_roc_auc(y_test, predict(rf_clf, type = "prob", X_test)[,2])

# -----------
library(pROC)
library(randomForest)

fraud.rf = randomForest(fraud~ age + gender + category + amount, data= train, ntree = 100, mtry=2, importance=TRUE, proximity=TRUE)
print(fraud.rf)
#----------------
# Decision tree
#----------------
# Classification

library("rpart") 
library("rpart.plot")
decision_tree <- rpart(fraud ~ step + age + gender + category + amount, method="class", data=fr_sup, control=rpart.control(minsplit=1), parms=list(split="information"))
summary(decision_tree)
rpart.plot(decision_tree, type=2, extra=1)

# #Naive Bayes
# library("e1071") 
# 
# fr_sup$amount1<-ifelse(fr_sup$amount<186,"<186",">186")
# training_data <- as.data.frame(fr_sup[1:dim(fr_sup)[1]-1,]) 
# test_data <- as.data.frame(fr_sup[dim(fr_sup)[1],])
# test_data
# 
# model1 <- naiveBayes(fraud ~ step + age + gender + category + amount1, training_data)
# model1
# 
# # Predict with test_data
# results1 <- predict(model1,test_data)
# results1



