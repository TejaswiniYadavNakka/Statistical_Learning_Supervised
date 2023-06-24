install.packages("stringr")
install.packages("dplyr")
install.packages("mice")
install.packages("ggplot2")
install.packages("caret")
install.packages("randomForest")
install.packages("xgboost")


library("stringr")
library("dplyr")
library("mice")
library("ggplot2")
library("caret")
library("randomForest")
library("xgboost")


#Loading the data
df <- read.csv("C:/Users/Tejaswini yadav/Desktop/Supervised learning/eda_data.csv")
head(df)
str(df)
colnames(df)
#Cleaning the data
df <- df %>% 
  dplyr::select(1,9,12,14,20,24:30)
summary(df)
colnames(df)
str(df)
head(df)

#Data cleaning(Age)
#Assuming People who are of age (18-70) are the working age group
boxplot(df$age, horizontal= T,main="BOXPLOT FOR AGE")

#Checking for people who do no belong to the working age
min_age <- sum(df$age < 18)
min_age
max_age <- sum(df$age > 70)
max_age

# To replace <18 outliers with IQR Q1 (30) & replace >70 outliers with IQR Q3(52)
getWorkingAge <- df$age[df$age <66 & df$age > 23 & !is.na(df$age)]
quantile(getWorkingAge)

df$age[df$age<18] <- 30
df$age[df$age>70] <- 52
hist(df$age,xlab="age",main="Age")

#Data Cleaning(Average Salary)
head(df$avg_salary)
df$avg_salary <- df$avg_salary*1000
head(df$avg_salary)
#Data Cleaning(Job Title Simplified)
# Replace NA with other tech jobs 
df$job_simp <- str_replace_all(df$job_simp, "\\bna\\b", "Other tech jobs")

#Data Cleaning(Categorizing number of employees in the size of company)
#1 to 200 employees - “Small”
#201 to 1000 employees - ‘Medium’
#1001 and above - ‘Large’

df$Size <- ifelse(df$Size %in% c("1 to 50 employees", "51 to 200 employees"), "small", 
                  ifelse(df$Size %in% c("201 to 500 employees", "501 to 1000 employees"), "medium", 
                         ifelse(df$Size %in% c("1001 to 5000 employees", "5001 to 10000 employees", "10000+ employees"), "large", NA)))


#Data Cleaning(Revenue categorized by size of the business)

#Less than $1 million (USD) - ‘micro-business’
#$1 million to 10 million (USD) - ‘small-business’
#$10 million to 50 million (USD) - ‘medium-business’
#$50 million and above (USD) - ‘large-business’


df$Revenue<- ifelse(df$Revenue == "Less than $1 million (USD)", "micro-business", 
                    ifelse(df$Revenue %in% c("$1 to $5 million (USD)", "$5 to $10 million (USD)"), "small-business", 
                           ifelse(df$Revenue %in% c("$10 to $25 million (USD)","$25 to $50 million (USD)", "$50 to $100 million (USD)"), "medium-business", 
                                  ifelse(df$Revenue %in% c("$100 to $500 million (USD)", "$500 million to $1 billion (USD)", "$1 to $2 billion (USD)", "$2 to $5 billion (USD)", "$5 to $10 billion (USD)", "$10+ billion (USD)"), "large-business", NA))))



#Data Cleaning - Impute Missing Value Revenue(195) and Size(2) by MICE. polyreg method was chosen because there are more than 2 factor variables.

df <- df %>% 
  mutate(
    Size = as.factor(Size),
    Revenue = as.factor(Revenue)
  )

init = mice(df, maxit = 0)

meth = init$method
predM = init$predictorMatrix
meth[c("Size")] = "polyreg"
meth[c("Revenue")] = "polyreg"

imp_rev <- mice(df, m=5, method= meth, predictorMatrix = predM, maxit = 10, seed = 20)

df <- complete(imp_rev)

#Data Cleaning(Industry)
df <- subset(df, Industry != "-1")


#Exploratory Data Analysis

#Relationship between age and average salary
ggplot(data = df, mapping = aes(x = age, y = avg_salary ))+labs(title="Age vs AverageSalary") + geom_point()

#comparing salary and company size
ggplot(data = df, aes(x = Size, y = avg_salary))+labs(title="CompanySize vs AverageSalary") + geom_boxplot()


#comparing salary distribution for each tech job
ggplot(data = df, aes(x = job_simp, y = avg_salary)) +labs(title="Tech Job vs AverageSalary") + geom_boxplot()




#job skills needed for tech job positions
skill <- c("python_yn", "R_yn","spark","aws","excel")
NoYes <- c("1","0")
df2 <- df %>% 
  dplyr::select('python_yn':'job_simp')
df2_new <- cbind(df2, skill)
df2_new2 <- cbind(df2_new, NoYes)

ggplot(df2_new2) + geom_bar(aes(x=skill, fill=NoYes),position = "dodge") + facet_wrap(~job_simp) +theme(axis.text.x=element_text(angle=30,hjust=0.5,vjust=0.5))


#Salary Prediction
#Feature Selection and Normalization
sal_df<-
  dplyr::select(df,-c("X","Industry"))
norm_minmax <- function(x){
  (x- min(x)) /(max(x)-min(x))
}
sal_df[sapply(sal_df, is.numeric)] <- lapply(sal_df[sapply(sal_df, is.numeric)],norm_minmax)
sal_df[sapply(sal_df, is.character)] <- lapply(sal_df[sapply(sal_df, is.character)],as.factor)
str(sal_df)


#Splitting the data into train 20% and test 80%
set.seed(20)
indice<-caret::createDataPartition(y=sal_df$avg_salary,p=0.8,list=FALSE)
sal_train<-sal_df[indice,]
sal_test<-sal_df[-indice,]

sal_train_x = sal_train[, -4]
sal_train_y = sal_train[,4]
sal_test_x = sal_test[,-4]
sal_test_y = sal_test[,4]

#Multiple Linear Regression model
# Create Multiple Regression Model
set.seed(20)
slmModel=lm(avg_salary~.,data=sal_train)
summary(slmModel)

# Predict Salary
pred_sal_slm<-predict(slmModel,sal_test_x)

# Evaluate Model
mse <- mean((sal_test_y - pred_sal_slm)^2)
mae <- MAE(sal_test_y, pred_sal_slm)
rmse <-RMSE(sal_test_y,pred_sal_slm)
r2 <- R2(sal_test_y, pred_sal_slm)
model_metrics_lm <- cbind(mse,mae,rmse,r2)
row.names(model_metrics_lm)<-"Multiple Linear Regression"
model_metrics_lm
overall<-rbind(overall,model_metrics_lm)


#Random Forest
#Create Random Forest Model
set.seed(20)
RFModel = randomForest(x = sal_train_x,
                       y = sal_train_y,
                       ntree = 500)

#Predict Salary
pred_sal_RF<-predict(RFModel,sal_test_x)
#Evaluate Model
mse = mean((sal_test_y - pred_sal_RF)^2)
mae = MAE(sal_test_y , pred_sal_RF)
rmse =RMSE(sal_test_y ,pred_sal_RF)
r2 = R2(sal_test_y, pred_sal_RF)
model_metrics_RF <- cbind(mse,mae,rmse,r2)
row.names(model_metrics_RF)<-"random forest"
model_metrics_RF
overall<-rbind(model_metrics_lm,model_metrics_RF)



#XGBoost
set.seed(20)
xgb_train = xgb.DMatrix(data = data.matrix(sal_train_x), label = sal_train_y)
xgb_test = xgb.DMatrix(data = data.matrix(sal_test_x), label = sal_test_y)
xgboostModel = xgboost(data = xgb_train, max.depth = 3, nrounds = 100, verbose = 0)

#Predict Salary
pred_sal_xgb = predict(xgboostModel, xgb_test)
#Evaluate Model
mse = mean((sal_test_y - pred_sal_xgb)^2)
mae = MAE(sal_test_y , pred_sal_xgb)
rmse =RMSE(sal_test_y ,pred_sal_xgb)
r2 = R2(sal_test_y, pred_sal_xgb)
model_metrics_xgb <- cbind(mse,mae,rmse,r2)
row.names(model_metrics_xgb)<-"XGBoost"
model_metrics_xgb
overall<-rbind(overall,model_metrics_xgb)
#Evaluation of regression models
overall




