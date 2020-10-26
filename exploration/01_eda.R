
library(dplyr)
library(ggplot2)

train <- read.csv("data/train.csv")

train$signal_fctr <- as.factor(train$signal)

summarise_signal <- function(df, ...){
  df %>% 
    summarise(
      n_signal = sum(signal),
      n = n(),
      sgn_ratio = sum(signal) / n(),
      ...
      )
  
}


train$B_PT_x <- train$B_PT * train$B_DIRA_OWNPV
train$B_PT_y <- train$B_PT * sqrt(1 - train$B_DIRA_OWNPV**2)

train %>% summarise_signal()

train %>% group_by(Kplus_P > 5e4) %>% summarise_signal()


train %>% 
  ggplot(aes(x = Kplus_P, color = signal_fctr, fill = signal_fctr)) + 
  geom_density(alpha = 0.3) + 
  geom_rug()

summary(train)  

cor(train %>% select(where(is.numeric)))
train %>% select(where(is.numeric))
# train %>% select(starts_with("a"), signal, where(is.numeric))
train %>% select(starts_with("Kplus")) %>% head()
train %>% select(starts_with("pi")) %>% head()

corrplot::corrplot(cor(train %>% select_if(is.numeric)))
