---
title: "testing xgb"
author: "brian chase"
date: "12/12/2020"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
require(xgboost)
```

## R Markdown


```{r cars}
library(xgboost)
# load data
data(agaricus.train, package = 'xgboost')
data(agaricus.test, package = 'xgboost')
train <- agaricus.train
test <- agaricus.test
# fit model
bst <- xgboost(data = train$data, label = train$label, max_depth = 2, eta = 1, nrounds = 2,nthread = 2, objective = "binary:logistic")
# predict
pred <- predict(bst, test$data)

```

