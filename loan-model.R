train <- read.csv("loan.train.csv", na.strings = c(""," ",NA))
test <- read.csv("loan.csv", na.strings = c(""," ",NA))

#visualize data
hist(train$ApplicantIncome, breaks = 300)
hist(train$CoapplicantIncome, breaks = 100)
boxplot(train$CoapplicantIncome)
boxplot(train$ApplicantIncome)
boxplot(train$LoanAmount)

#Clean up data
train$Credit_History <- as.factor(train$Credit_History)
test$Credit_History <- as.factor(test$Credit_History)

train$CoapplicantIncome <- as.integer(train$CoapplicantIncome)
test$CoapplicantIncome <- as.integer(test$CoapplicantIncome)

levels(train$Dependents)[4] <- "3"
levels(test$Dependents)[4] <- "3"

test$Loan_Status <- sample(0:1, size = 367, replace = T)
test$Loan_Status <- ifelse(test$Loan_Status == 0, "N", "Y")
test$Loan_Status <- factor(test$Loan_Status, levels = c("N", "Y"))

#Fill missing values for credit_history
train$Credit_History[c(280,126,324,545,319,220,238,474,199,87,261,80,396,349,461,498,445,188,507,492,504,531,314,310,31,452,450,118,557,318,364,393,491,157,260,17,131,43,378,412,566)] <- 1
train$Credit_History[c(601,130, 84,182,25,584,96,237,534)] = 0

test$Credit_History[c(4,13,27,100,140,46,144,140,180,221,263,283, 287,330,337,359,361,365,91,178)] <- 1
test$Credit_History[c(29,105,116,165,203,260,266,306,352,186)] <- 0

#Combine datasets for imputing missing values
combi <- rbind(train, test)
str(combi)
summary(combi)

#Check for proportion of missing values
pMiss <- function(x){sum(is.na(x))/length(x)*100}
apply(combi,2,pMiss)
apply(combi,1,pMiss)

library(mice)
md.pattern(combi)

#create a copy
a <- combi

combi$Loan_Status <- NULL

#Separate variables based on classes for imputation
az <- split(names(combi), sapply(combi, function(x){class(x)}))
bin.fact <- combi[az$factor]
mult.fact <- bin.fact[,c(4,8)] #factor with multiple levels
bin.fact <- bin.fact[,-c(1,4,8)] #factor with binary levels
int <- combi[az$integer] #integers

temp <- mice(bin.fact, m=5, maxit=50, method = "logreg", seed = 500)
temp.mult <- mice(mult.fact, maxit = 50, method = "polyreg", seed = 500)
temp.num <- mice(int, maxit = 50, method = "pmm", seed = 500)

summary(temp)

bin <- complete(temp, 4)
mult <- complete(temp.mult, 3)
int <- complete(temp.num,3)

combi <- cbind(bin, mult, int)
combi$Loan_ID <- a$Loan_ID
combi$Loan_Status <- a$Loan_Status

library(mlr)
cd <- capLargeValues(combi, target = "Loan_Status", cols = c("ApplicantIncome"), threshold = 40000)
cd <- capLargeValues(cd, target = "Loan_Status", cols = c("CoapplicantIncome"), threshold = 16000)
cd <- capLargeValues(cd, target = "Loan_Status", cols = c("LoanAmount"), threshold = 520)

#feature engineering
#Income by loan
cd$Loan_by_Income <- with(cd, (LoanAmount*1000)/Loan_Amount_Term)


new_train <- cd[1:nrow(train),]
new_test <- cd[-(1:nrow(train)),]
str(new_train)

new_train$Loan_Status <- ifelse(new_train$Loan_Status == "N", 0, 1)
new_train$Loan_Status <- factor(new_train$Loan_Status, levels = c(0,1))

new_train$Loan_ID <- NULL
new_test$Loan_ID <- NULL

#MODEL BUILDING - LOGISTIC REGRESSION
mod <- glm(Loan_Status~Married+Education+Property_Area+LoanAmount*Self_Employed+Dependents*ApplicantIncome+Credit_History+Loan_by_Income+Credit_History*CoapplicantIncome,
           data=new_train, family = binomial)

summary(mod)
anova(mod, test = "Chisq")

#Make predictions
pred <- predict(mod, newdata = new_test[,-12], type = "response")
pred <- ifelse(pred > 0.65, "Y", "N")

glm.loan <- data.frame(Loan_ID = test$Loan_ID,
                        Loan_Status = pred)

write.csv(glm.loan, "glm.loan.csv", row.names = F)

#Accuracy - 0.796, Rank - 81
