# abb 2/4/2020: looking for good imbalanced example
#source('SafeDriver2.R',echo=T,max.deparse.length=Inf)
basefile='SafeDriver2'
Sys.umask( '0002' )
set.seed(1, "L'Ecuyer-CMRG") # reproducible, multicore
dir.create( 'logs', showWarnings=F, recursive=T )
logfile <- file.path('logs', basefile)
sink( file=logfile, split=T )
print(Sys.time())

#https://www.kaggle.com/captcalculator/r-xgboost-with-caret-tuning-and-gini-score
# This is a minimal framework for training xgboost in R using caret to do the cross-validation/grid tuning
# and using the normalized gini metric for scoring. The # of CV folds and size of the tuning grid
# are limited to remain under kaggle kernel limits. To improve the score up the nrounds and expand
# the tuning grid.
#https://www.kaggle.com/kueipo/base-on-froza-pascal-single-xgb-lb-0-284 Private Score 0.28594 Public Score 0.28425 based on:
#1. https://www.kaggle.com/abhilashawasthi/forza-baseline Private Score 0.28164 Public Score 0.27999
#2. https://www.kaggle.com/pnagel/reconstruction-of-ps-reg-03 (amazing detective work)
#https://www.kaggle.com/nigelcarpenter/r-xgboost-with-gini-eval-and-stopping
#https://www.kaggle.com/headsortails/steering-wheel-of-fortune-porto-seguro-eda/code
#https://www.kaggle.com/captcalculator/r-xgboost-with-caret-tuning-and-gini-score
#https://www.kaggle.com/nitinakaggle/xgboost-for-porto-seguro-s-safe-driver-prediction
#https://www.kaggle.com/nitinakaggle/porto-seguro-s-insurance-prediction-using-xgboost

library(data.table)
library(caret)
library(xgboost)

# Read train and test data
#dtrain <- fread('../input/train.csv')
#dtest <- fread('../input/test.csv')
#system.time(print(load( 'SafeDriver.RData' ))) # 2s, 29mb, sd0.train, sd0.test
system.time(print(load( 'SafeDriverTrain.RData' ))) # 1s, 12mb, sd0.train
system.time(print(load( 'SafeDriverTest.RData' ))) # 1s, 18mb, sd0.test, sd0.submission

# "Values of -1 indicate that the feature was missing" https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data
min(sd0.train[,-1]) # -1
tail(round(100*sort(sapply( sd0.train, function(x) sum(x==-1) ))/nrow(sd0.train),1),5)
tail(round(100*sort(sapply( sd0.test, function(x) sum(x==-1) ))/nrow(sd0.test),1),5)
#ps_car_07_cat     ps_car_14     ps_reg_03 ps_car_05_cat ps_car_03_cat 
#          1.9           7.2          18.1          44.8          69.1
#ps_car_07_cat     ps_car_14     ps_reg_03 ps_car_05_cat ps_car_03_cat 
#          1.9           7.1          18.1          44.8          69.1
# simple imputation with median:
for( i in which(sapply( sd0.train, function(x) sum(x==-1) ) > 0) ) {
    x <- sd0.train[[i]]
    sd0.train[[i]][x == -1] <- median( x[x!=-1] )
}
for( i in which(sapply( sd0.test, function(x) sum(x==-1) ) > 0) ) {
    x <- sd0.test[[i]]
    sd0.test[[i]][x == -1] <- median( x[x!=-1] )
}
stopifnot( sd0.train[,-1] >= 0 )

# abb1: drop _calc_* fields (per discussion, reason unknown?)
dtrain <- data.table(sd0.train[, grep( '_calc_', colnames(sd0.train), value=T, invert=T )])
dtest <- data.table(sd0.test[, grep( '_calc_', colnames(sd0.test), value=T, invert=T )])
# collect names of all categorical variables
(cat_vars <- names(dtrain)[grepl('_cat$', names(dtrain))])
# turn categorical features into factors
dtrain[, (cat_vars) := lapply(.SD, factor), .SDcols = cat_vars]
dtest[, (cat_vars) := lapply(.SD, factor), .SDcols = cat_vars]

# one hot encode the factor levels
dtrain <- as.data.frame(model.matrix(~. - 1, data = dtrain))
dtest <- as.data.frame(model.matrix(~ . - 1, data = dtest))

# create index for train/test split
set.seed(1, "L'Ecuyer-CMRG") # reproducible, multicore
train_index <- sample(c(TRUE, FALSE), size = nrow(dtrain), replace = TRUE, prob = c(0.8, 0.2))

# perform x/y ,train/test split.
x_train <- dtrain[train_index, -(1:2)]
y_train <- as.factor(dtrain$target[train_index])

x_test <- dtrain[!train_index, 3:ncol(dtrain)]
y_test <- as.factor(dtrain$target[!train_index])

# Convert target factor levels to 0 = "No" and 1 = "Yes" to avoid this error when predicting class probs:
# https://stackoverflow.com/questions/18402016/error-when-i-try-to-predict-class-probabilities-in-r-caret
levels(y_train) <- c("No", "Yes")
levels(y_test) <- c("No", "Yes")

# normalized gini function taked from:
# https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703
normalizedGini <- function(aa, pp) {
    Gini <- function(a, p) {
        if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
        temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
        temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
        population.delta <- 1 / length(a)
        total.losses <- sum(a)
        null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
        accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
        gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
        sum(gini.sum) / length(a)
    }
    Gini(aa,pp) / Gini(aa,aa)
}

# create the normalized gini summary function to pass into caret
giniSummary <- function (data, lev = "Yes", model = NULL) {
    levels(data$obs) <- c('0', '1')
    out <- normalizedGini(as.numeric(levels(data$obs))[data$obs], data[, lev[2]])  
    names(out) <- "NormalizedGini"
    out
}

grep('xgb',names(getModelInfo()),ignore.case=T,value=T) # "xgbDART"   "xgbLinear" "xgbTree"
grep('gbm',names(getModelInfo()),ignore.case=T,value=T) # "FH.GBML" "gbm_h2o" "gbm"
grep('bart',names(getModelInfo()),ignore.case=T,value=T) # "bartMachine"
paste( modelLookup("xgbTree")$parameter, collapse=', ' ) # nrounds, max_depth, eta, gamma, colsample_bytree, min_child_weight, subsample

# crudely estimate per-iteration tuning time with simplest setup + major assumptions.
tc0 <- trainControl( number=3, method='cv', summaryFunction=giniSummary, allowParallel=T, classProbs=F, verboseIter=T )
tg0 <- tg1 <- data.frame( nrounds=50, max_depth=6, eta=0.3, gamma=0.1, min_child_weight=5, subsample=1, colsample_bytree=1 )
print(itertime <- system.time(m1 <- train( x=x_train, y=y_train, method='xgbTree', trControl=tc0, tuneGrid=tg0, metric='NormalizedGini' )))
(itertime=itertime['elapsed']/(tc0$number*tg0$nrounds)) # seconds per iter, per round

# training control for tuning with grid search
#xgboost: the seed object should be a list of length 11 with 10 integer vectors of size 36 and the last list element having at least a single integer set.seed(1); seeds <- sample(1e6,1e4) # reproducible
tc2 <- trainControl( # instructions to caret
    method='cv',
    number=3, # folds/resamples, >5
    summaryFunction=giniSummary,
    allowParallel=T,
    search='grid', # grid|random
    classProbs=T,
#    sampling='smote',
    verboseIter=T
)

# theory: hyperparameters interact in clusters. set a few that are more independent.
# clusters: 1. colsample_bytree+subsample+gamma, 2. eta+max_depth+min_child_weight+nrounds

if( F ) {
# 1. ballpark complexity first: max_depth+min_child_weight
tg <- expand.grid( max_depth=seq(3,9,by=3), min_child_weight=seq(1, 201, by=40) )
#m2a=m2; tg <- expand.grid( max_depth=4:7, min_child_weight=201 )
tg <- cbind( tg1[,setdiff(colnames(tg1),colnames(tg))], tg )
(itertime*tc2$number*sum(tg$nrounds))/(60) # estimated minutes
system.time(print(m2 <- train( x=x_train, y=y_train, method='xgbTree', trControl=tc2, tuneGrid=tg, metric='NormalizedGini' )))
m2$results[which.max(m2$results$NormalizedGini),c('max_depth','min_child_weight','NormalizedGini')]
#m2a$results[which.max(m2a$results$NormalizedGini),c('max_depth','min_child_weight','NormalizedGini')]
plot(m2) # clearly best: depth~5; not as important: minweight~120 (100-200, err lower)
tg1$max_depth=5; tg1$min_child_weight=120

# 2. ballpark over/under-fit bias-variance next: eta+nrounds
#nr=1e4; set.seed(1); i=sample(nrow(x_train), nr) # crude ranges from sample
#tg <- expand.grid( eta=seq(0.01,0.901,length=3), nrounds=seq(30, 1000, length=3) )
nr=nrow(x_train); i=1:nrow(x_train)
tg <- expand.grid( eta=seq(0.01,0.101,length=5), nrounds=500 )
tg <- cbind( tg1[,setdiff(colnames(tg1),colnames(tg))], tg )
(tg$min_child_weight <- tg$min_child_weight*(length(i)/nrow(x_train)))
((nr/nrow(x_train))*itertime*tc2$number*sum(tg$nrounds[tg$nrounds==max(tg$nrounds)]))/(60) # estimated minutes
system.time(print(m2 <- train( x=x_train[i,], y=y_train[i], method='xgbTree', trControl=tc2, tuneGrid=tg, metric='NormalizedGini' )))
m2$results[which.max(m2$results$NormalizedGini),c('eta','nrounds','NormalizedGini')]
plot(m2) # clearly best: eta=0.03275
tg1$eta=0.03; tg1$nrounds=500

# 3. ballpark subsample+gamma (can't tune on colsample_by*?)
#https://www.kaggle.com/kueipo/base-on-froza-pascal-single-xgb-lb-0-284: 'colsample_bytree': 0.7, 'colsample_bylevel':0.7,
nr=nrow(x_train); i=1:nrow(x_train)
#tg <- expand.grid( nrounds=50, subsample=seq(0.5,1,length=3), gamma=seq(0,2,length=3) ) 
# probably want gamma=0, then increase if overfitting rears its ugly head.
tg <- expand.grid( nrounds=50, subsample=seq(0.1,0.5,length=5), gamma=0 )
tg <- cbind( tg1[,setdiff(colnames(tg1),colnames(tg))], tg )
tg$min_child_weight <- tg$min_child_weight*(length(i)/nrow(x_train))
((nr/nrow(x_train))*itertime*tc2$number*sum(tg$nrounds[tg$nrounds==max(tg$nrounds)]))/(60) # estimated minutes
tg
system.time(print(m2 <- train( x=x_train[i,], y=y_train[i], method='xgbTree', trControl=tc2, tuneGrid=tg, metric='NormalizedGini' )))
plot(m2) # clearly best: subsample=0.3
tg1$subsample=0.3; tg1$gamma=0

}

# critical for imbalanced datasets: split into 75/25 train/validation to check for overfit
set.seed(1); i.validation <- sort(sample( nrow(dtrain), 0.25*nrow(dtrain) )); i.train <- (1:nrow(dtrain))[-i.validation]
tg1$max_depth=5; tg1$min_child_weight=120
tg1$eta=0.03; tg1$nrounds=500
tg1$subsample=0.3; tg1$gamma=0

dtrain.xgb <- xgb.DMatrix( data=as.matrix(dtrain[train_index,-2]), label=as.integer(dtrain$target[train_index]==1) ) # include dtrain$id
dvalidate.xgb <- xgb.DMatrix( data=as.matrix(dtrain[!train_index,-2]), label=as.integer(dtrain$target[!train_index]==1) ) # include dtrain$id
Normalized_Gini_XGB = function( preds, train ){
    actual = getinfo( train, 'label')
    score = normalizedGini( actual, preds )
    return(list(metric = "NormalizedGini", value = score))
}

# 4. optimize nrounds & gamma
tc3 <- tc2
tc3$number <- 10 # lots of folds
tg <- expand.grid( nrounds=c(500,1000,1500), gamma=c(0,4,8,12,20,30) )
#debug: tc3$number <- 3; tg <- expand.grid( nrounds=10, gamma=0 )
tg <- cbind( tg1[,setdiff(colnames(tg1),colnames(tg))], tg )
nr=sum(train_index); i=which(train_index)
((nr/nrow(x_train))*itertime*tc3$number*sum(tg$nrounds[tg$nrounds==max(tg$nrounds)]))/(60) # estimated minutes
tg$min_child_weight <- tg$min_child_weight*(length(i)/nrow(x_train))
tg
set.seed(1, "L'Ecuyer-CMRG") # reproducible, multicore
#system.time(m3 <- train( x=x_train, y=y_train, method='xgbTree', trControl=tc3, tuneGrid=tg, metric='NormalizedGini', print_every_n=100 ))
system.time(print(m3 <- train( x=x_train, y=y_train, method='xgbTree', trControl=tc3, tuneGrid=tg, metric='NormalizedGini', print_every_n=100, feval=Normalized_Gini_XGB, maximize=T, watchlist=list(train=dtrain.xgb) )))

plot(m3) #
stop()
# abb6: 40926s, The final values used for the model were nrounds = 1500, max_depth = 5, eta = 0.03, gamma = 8, colsample_bytree = 1, min_child_weight = 120 and subsample = 0.3.
i.train=which(train_index)
i.validation=which(!train_index)


tp1 <- list( booster="gbtree", objective="binary:logistic", max_depth=tg1$max_depth, eta=tg1$eta, min_child_weight=tg1$min_child_weight, subsample=tg1$subsample )
tp1$colsample_bytree=tp1$colsample_bylevel=0.7
tp1$gamma=8 # too much:100

set.seed(1, "L'Ecuyer-CMRG") # reproducible, multicore
system.time(m4 <- xgb.train( data=dtrain.xgb, params=tp1, nrounds=1500, feval=Normalized_Gini_XGB, maximize=T, watchlist=list(train=dtrain.xgb), verbose=1, print_every_n=25 ))

# abb0: 12m, 8 iter @ 300 rounds, so ~90s/iter
# abb1: 5*6*3*5=450 iter * 90s=675m? kill at 17.5h
# abb2: 200s, gamma=0, colsamp*=0.7, ngini=0.313 (overfitting)
# abb3: 200s, gamma=10, colsamp*=0.7, ngini=0.288
# abb4: 200s, gamma=30, colsamp*=0.7, ngini=0.261
# abb5: 200s, gamma=20, colsamp*=0.7, ngini=0.270

# make predictions
p.train <- predict( m4, newdata=dtrain.xgb, type='prob' )
p.validation <- predict( m4, newdata=dvalidate.xgb, type='prob' )
normalizedGini( dtrain$target[i.train], p.train ); normalizedGini( dtrain$target[i.validation], p.validation )
# abb0: 0.2802011
# abb2: 0.3237133, 0.2795222
# abb3: 0.2928979, 0.2715281
# abb4: 0.2656059, 0.248704
# abb5: 0.2740145, 0.2558822
# abb6: 0.3184043, 0.27442

print(Sys.time())
if( sink.number() > 0 ) sink( file=NULL )

