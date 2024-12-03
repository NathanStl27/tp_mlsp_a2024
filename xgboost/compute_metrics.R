compute_metrics <- function(y_true, y_pred, itr=0){
  #-----------------------------------
  # * Matrice de confusion :
  # TN FP
  # FN TP
  #-----------------------------------
  #
  ## OUT
  # * Vecteur des mesures de performance
  #-----------------------------------
  y_true <- y_true + 1 # échelle de 1 à 5
  y_pred <- y_pred + 1 
  
  cm <- table(y_true, y_pred) # confusion matrix
  
  n <- sum(cm) # number of instances
  nc <- nrow(cm) # number of classes
  diag <- diag(cm) # number of correctly classified instances per class 
  rowsums <- apply(cm, 1, sum) # number of instances per class
  colsums <- apply(cm, 2, sum) # number of predictions per class
  p <- rowsums / n # distribution of instances over the actual classes
  q <- colsums / n # distribution of instances over the predicted classes
  
  cat("-- Accuracy stats -- \n")
  # Overall accuracy
  accuracy <- sum(diag) / n 
  cat("Overall accuracy : ", accuracy)
  
  # Average accuracy
  avgAccuracy <- diag/colsums
  cat("\nAverage accuracy : ", avgAccuracy)
  
  cat("\n\n-- Per-class metrics --\n")
  # Per-class precision/recall/F1
  precision <- diag / colsums 
  recall <- diag / rowsums 
  f1 <- 2 * precision * recall / (precision + recall) 
  pc_metrics <- data.frame("precision" = precision, "recall" = recall, "f1-score" = f1)
  print(pc_metrics)
  
  cat("\n-- Macro averaged metrics --\n")
  # Macro averaged metrics
  macroPrecision <- mean(precision)
  macroRecall <- mean(recall)
  macroF1 <- mean(f1)
  macro <- data.frame("precision" = macroPrecision, "recall" = macroRecall, "f1-score" = macroF1)
  print(macro)
  
  cat("\n-- Micro averaged metrics --\n")
  # Micro averaged metrics
  oneVsAll = lapply(1 : nc,
                    function(i){
                      v = c(cm[i,i],
                            rowsums[i] - cm[i,i],
                            colsums[i] - cm[i,i],
                            n-rowsums[i] - colsums[i] + cm[i,i]);
                      return(matrix(v, nrow = 2, byrow = T))})
  
  s = matrix(0, nrow = 2, ncol = 2)
  for(i in 1 : nc){s = s + oneVsAll[[i]]}
  
  micro_prf <- (diag(s) / apply(s,1, sum))
  cat("micro-precision and recall : ", micro_prf)
  
  cat("\n\n-- Random guess metrics --\n")
  # Random guess metrics
  rg <- (n / nc) * matrix(rep(p, nc), nc, nc, byrow=F)
  rgAccuracy <- 1 / nc
  rgPrecision <- p
  rgRecall <- 0*p + 1 / nc
  rgF1 <- 2 * p / (nc * p + 1)
  
  print("Expected confusion matrix : \n")
  print(rg)
  cat("\nRG accuracy : ", rgAccuracy, "\n")
  rg_df <- data.frame("precision" = rgPrecision, "recall" = rgRecall, "f1-score" = rgF1)
  print(rg_df)
  
  res <- list(
    "itr" = itr,
    "acc" = accuracy,
    "avg_acc" = avgAccuracy,
    "per_class" = pc_metrics,
    "macro" = macro,
    "micro" = micro_prf,
    "rg_acc" = rgAccuracy,
    "rg_df" = rg_df
  )
  
  return(res)
}


