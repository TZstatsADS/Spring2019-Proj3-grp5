########################
### Cross Validation ###
########################

### Author: Xinyi Hu
### Project 3


cv.function <- function(X.train, y.train, d, K){
  
  n <- dim(y.train)[1]
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))
  cv.error <- rep(NA, K)
  
  for (i in 1:K){
    train.data <- X.train[s != i, ,]
    train.label <- y.train[s != i, ,]
    test.data <- X.train[s == i, ,]
    test.label <- y.train[s == i, ,]

    # fit linear model
    cat("Fold =", i, "\n")
    par <- list(nr=d)
    fit <- train(train.data, train.label,par)
    #fit <- train_xgboost(train.data, train.label, par )
    
    cat(i, "Train finish","\n")
    pred <- test(fit,test.data)
    #pred <- test_xgboost(fit, test.data)  
    
    cv.error[i] <- mean((pred - test.label)^2)
    cat(cv.error[i], "\n" )
  }
  return(c(mean(cv.error),sd(cv.error)))
}

