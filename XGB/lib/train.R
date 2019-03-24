#########################################################
### Train a classification model with training features ###
#########################################################

### Author: Chengliang Tang
### Project 3


train = function(dat_train, label_train, par=NULL){
  
  ### Train a Gradient Boosting Model (GBM) using processed features from training images
  
  ### Input: 
  ###  -  features from LR images 
  ###  -  responses from HR images
  ### Output: a list for trained models
  
  ### load libraries
  library(gbm)
  library(xgboost)
  library(BayesTree)
  library(iRF)
  library(rpart)

  ### creat model list
  modelList = list()
  
  ### Train with XGboost
  if(is.null(par)){
    nr = 100
  } else {
    nr = par$nr
  }
  
  ### the dimension of response arrat is * x 4 x 3, which requires 12 classifiers
  ### this part can be parallelized
  for (i in 1:12){
    ## calculate column and channel
    c1 = (i-1) %% 4 + 1
    c2 = (i-c1) %/% 4 + 1
    featMat = dat_train[, , c2]
    labMat = label_train[, c1, c2]
    
    ## fit the model
    cat("Classifer =", i, "\n")
    train.xgb = xgb.DMatrix(featMat, label =labMat)
    #xgb_model = xgboost(data = train.xgb, booster = "gblinear", seed = 1, nrounds = nr, verbose = FALSE, 
     #                      objective = "reg:linear", eval_metric = "rmse", lambda = 1, 
      #                     alpha = 0)
    xgb_model = xgboost(data = train.xgb,booster = "gblinear", seed = 1, nrounds = nr,  
                        objective = "reg:linear", eval_metric = "rmse", lambda = 1, 
                        alpha = 0,nfold =3,max_depth = 6, min_child_weight =6,gamma= 0.2, print_every_n = FALSE)
    
    modelList[[i]] = list(fit=xgb_model, iter=par)
  }
  return(modelList)
}

