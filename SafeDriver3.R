# abb 2/4/2020: looking for good imbalanced example
# https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
# this approximates https://www.kaggle.com/abhilashawasthi/forza-baseline Private Score 0.28164 Public Score 0.27999
#see also: https://github.com/Zindear/Applied-Multivariate-Data-Analysis
#source('SafeDriver3.R',echo=T,max.deparse.length=Inf)

require(xgboost) # otherwise base R
basefile='SafeDriver3a' # just for keeping track of iterations
Sys.umask( '0002' )
dir.create( 'logs', showWarnings=F, recursive=T )
logfile <- file.path('logs', basefile)
sink( file=logfile, split=T )
print(Sys.time())

#wget https://raw.githubusercontent.com/brunocampos01/porto-seguro-safe-driver-prediction/master/data/raw/datasets.zip; unzip datasets.zip # read into R and save xz-compressed RData file. single file too big for github, splitting ...
system.time(print(load( 'SafeDriverTrain.RData' ))) # 1s, 12mb, sd0.train
system.time(print(load( 'SafeDriverTest.RData' ))) # 1s, 18mb, sd0.test, sd0.submission
x.train <- as.matrix( sd0.train[,-2,drop=F] ) # including id, can be predictive
prop.table(table( y.train <- sd0.train[,2], useNA='always' )) # balance=4/96%

# create index for train/validation indicator
set.seed(1, "L'Ecuyer-CMRG") # reproducible, multicore
prop.table(table(itrain <- sample( c(T,F), size=nrow(x.train), replace=T, prob=c(0.75,0.25) )))

#require(MLmetrics) # for NormalizedGini(), gini=2*AUC−1, maps AUC to [0,1]
NormalizedGini <- function (y_pred, y_true) {
    SumGini <- function(y_pred, y_true) {
        y_true_sort <- as.numeric(y_true)[order(y_pred, decreasing=T)]
        y_Lorentz <- cumsum(y_true_sort)/sum(y_true_sort)
        y_random <- 1:length(y_pred)/length(y_pred)
        return(sum(y_Lorentz - y_random))
    }
    return(NormalizedGini <- SumGini(y_pred, y_true)/SumGini(y_true, y_true))
}
NormalizedGiniXGB = function( y_pred, y_train ){
    y_true = getinfo( y_train, 'label')
    score = NormalizedGini( y_pred, y_true )
    return(list( metric="ngini", value=score ))
}

# XGBoost
d1 <- xgb.DMatrix(x.train[itrain,], label=y.train[itrain]) # training frame
d2 <- xgb.DMatrix(x.train[!itrain,], label=y.train[!itrain]) # validation frame
#runlabel=paste0(basefile, '_run0')
# params from https://www.kaggle.com/abhilashawasthi/forza-baseline Private Score 0.28164 Public Score 0.27999
xgb.param1 <- list( eta=0.02, max_depth=4, subsample=0.9, colsample_bytree=0.9, objective='binary:logistic', seed=99, silent=T )
# run1: try params from my tuning
#runlabel=paste0(basefile, '_run1')
#xgb.param1 <- list( eta=0.03, max_depth=5, subsample=0.3, colsample_bytree=0.9, gamma=8, min_child_weight=120, objective='binary:logistic', seed=99, silent=T )
# run2: try smote (doesn't do anything here?) list( sample='smote', ...)

set.seed(1, "L'Ecuyer-CMRG") # reproducible, multicore
system.time(mx1 <- xgb.train( xgb.param1, d1, nrounds=5000, feval=NormalizedGiniXGB, maximize=T, watchlist=list(train=d1, valid=d2), early_stopping_rounds=100, print_every_n=100 )) #verbose_eval=100 ))
# forza: Best iteration: [759] train-gini:0.349262 valid-gini:0.277611
# run0 from raw data: Best iteration: [685]   train-ngini:0.345306    valid-ngini:0.278298
# run1: 200s, Best iteration: [667]   train-ngini:0.326298    valid-ngini:0.278089
# run2: 186s, Best iteration: [685]   train-ngini:0.345306    valid-ngini:0.278298

stopifnot(!is.na(l1 <- match( sd0.test$id, sd0.submission$id ))) # table(diff(l1))==1, just checking
sd0.submission$target[l1] <- predict( mx1, xgb.DMatrix(as.matrix(sd0.test)), ntree_limit=mx1$best_ntreelimit+50 )
write.csv( sd0.submission, paste0(runlabel,'.csv'), quote=F, row.names=F )
summary( sd0.submission )
# run0 from raw data:
#       id              target        
# Min.   :      0   Min.   :0.008626  
# 1st Qu.: 372022   1st Qu.:0.024335  
# Median : 744307   Median :0.032311  
# Mean   : 744154   Mean   :0.036672  
# 3rd Qu.:1116308   3rd Qu.:0.043319  
# Max.   :1488026   Max.   :0.712561

#wget -O SafeDriverForzaSubmission.csv 'https://www.kaggleusercontent.com/kf/1560289/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..XbCGxC5g1DcEPkF841adXg.ZOK0clRpVXRIXNOwVj_Zvn8q6um_4gflCD76QUcKW1cU6qFXV8x_OtCzQd_DV4i9zPyZ66vLdnhuM4eoV4WdfOz57pilQXG2Ql_jFTRNdUN58kpwdP1LbqyHNhX16qLuMFUQYH6k8auStwcSHjVqSw.HbFLbnzd8_4cebI9X8_fBg/submission.csv'
fs1 <- read.csv( 'SafeDriverForzaSubmission.csv.xz' )
sqrt(mean(( fs1$target - sd0.submission$target )^2))
NormalizedGini( sd0.submission$target, fs1$target )
# run0 from raw data: 0.004959259, 0.9785352
# run1 from raw data: 0.007237502, 0.967153

print(Sys.time())
if( sink.number() > 0 ) sink( file=NULL )
