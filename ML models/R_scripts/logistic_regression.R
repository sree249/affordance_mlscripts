
library(caret)

data_x <- read.csv('~/Dropbox/SYR_GAship/afforadance_Study/ML models/interaction_exploration/Smart_Phone_matrices_important.csv')
data <- read.csv('~/Dropbox/SYR_GAship/afforadance_Study/ML models/interaction_exploration/Smart_Phone_matrices_important_all.csv')
data_y <- read.csv('~/Dropbox/SYR_GAship/afforadance_Study/ML models/interaction_exploration/Smart_Phone_matrices_y_important.csv')
data_x <- read.csv('~/Dropbox/SYR_GAship/afforadance_Study/ML models/interaction_exploration/Smart_Phone_matrices.csv')

data$actual_use = data_y$actual_use

#flattenCorrMatrix <- function(cormat, pmat) {
#  ut <- upper.tri(cormat)
#  data.frame(
#    row = rownames(cormat)[row(cormat)[ut]],
#    column = rownames(cormat)[col(cormat)[ut]],
#    cor  =(cormat)[ut],
#    p = pmat[ut]
#  )
#}

library(Hmisc)
res2<-rcorr(as.matrix(data_x),type="pearson")
dim(res2$P)

table(res2$P<0.05)

correlationMatrix <- cor(as.matrix(data_x))

highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
names = colnames(data_x)[-highlyCorrelated]

data_test.actual_use = data['actual_use']
data_test = data[names]

data_test$actual_use = data$actual_use

Train <- createDataPartition(data_test, p=0.6, list=FALSE)
training <- data[ Train, ]
testing <- data[-Train, ]

mylogit <- glm(actual_use ~ .,data = training, family = "binomial"(link='logit'))

summary(mylogit)

data_test.prob <- predict(mylogit,data_test, type = "response")

data_test$pred <- 1

data_test$pred[data_test.prob < .5] = 0

table(data_test$pred, data_test$actual_use)



test = rcorr(c(1,0,1,0,1,0),c(0,1,0,1,0,1),type="pearson")
test$r
flattenCorrMatrix(res2$r, res2$P)

ut <- upper.tri(res2$r[res2$P<0.05])
table(res2$P<0.05)
dim(res2$P)

f <- function(x) {
  length(x[x>.75])
}

res2$r[,'Smart_Phone_Q10_feat3']
res2$P[is.na(res2$P)] <- 1

table1 <- apply(res2$r, 1, f)
table1[table1<70]
dim(table1)


findCorrelation(data_x, cutoff = 0.9, exact = TRUE)

Train <- createDataPartition(data$actual_use, p=0.6, list=FALSE)
training <- data[ Train, ]
testing <- data[-Train, ]

mean(training$actual_use)
mean(testing$actual_use)


mylogit <- glm(actual_use ~ .,data = training, family = "binomial"(link='logit'))

summary(mylogit)

testing.prob <- predict(mylogit,testing, type = "response")

testing$pred <- 1

testing$pred[testing.prob < .5] = 0

table(testing$pred, testing$actual_use)


mean((testing$actual_use - newdata_y)^2)
mean(as.integer(data_y$actual_use))
