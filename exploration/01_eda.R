
library(dplyr)
library(ggplot2)
library(yardstick)

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

names(train)

train$B_PT_x <- train$B_PT * train$B_DIRA_OWNPV
train$B_PT_y <- train$B_PT * sqrt(1 - train$B_DIRA_OWNPV**2)

train <- train %>% 
  mutate(
    P_diff_x = Kplus_P * Kst_892_0_cosThetaH + piminus_P * Kst_892_0_cosThetaH + gamma_PT - B_PT,
    P_diff_x0 = Kplus_P * Kst_892_0_cosThetaH + piminus_P * Kst_892_0_cosThetaH - gamma_PT - B_PT,
    P_diff_x01 = Kplus_P * Kst_892_0_cosThetaH - piminus_P * Kst_892_0_cosThetaH + gamma_PT - B_PT,
    P_diff_x1 = Kplus_P * Kst_892_0_cosThetaH - piminus_P * Kst_892_0_cosThetaH + gamma_PT - B_PT,
    P_diff_x2 = - Kplus_P * Kst_892_0_cosThetaH + piminus_P * Kst_892_0_cosThetaH + gamma_PT - B_PT,
    P_diff_y = Kplus_P * sqrt(1 - Kst_892_0_cosThetaH**2) + piminus_P * sqrt(1 - Kst_892_0_cosThetaH**2) + gamma_PT - B_PT,
    P_diff_y1 = - Kplus_P * sqrt(1 - Kst_892_0_cosThetaH**2) + piminus_P * sqrt(1 - Kst_892_0_cosThetaH**2) + gamma_PT - B_PT,
    P_diff_y2 = Kplus_P * sqrt(1 - Kst_892_0_cosThetaH**2) - piminus_P * sqrt(1 - Kst_892_0_cosThetaH**2) + gamma_PT - B_PT,
    P_diff_sq =  Kplus_P**2 + piminus_P**2 + gamma_PT**2 - B_PT**2
  )

train %>% 
  ggplot(aes(x = P_diff_sq, color = signal_fctr, fill = signal_fctr)) + 
  geom_density(alpha = 0.3) + 
  geom_rug()

train %>% 
  ggplot(aes(x = P_diff_x0, color = signal_fctr, fill = signal_fctr)) + 
  geom_density(alpha = 0.3) + 
  geom_rug()


train %>% 
  ggplot(aes(x = P_diff_x, color = signal_fctr, fill = signal_fctr)) + 
  geom_density(alpha = 0.3) + 
  geom_rug()

train %>% 
  ggplot(aes(x = P_diff_y, color = signal_fctr, fill = signal_fctr)) + 
  geom_density(alpha = 0.3) + 
  geom_rug()


train %>% 
  ggplot(aes(x = P_diff_x1, color = signal_fctr, fill = signal_fctr)) + 
  geom_density(alpha = 0.3) + 
  geom_rug()

train %>% 
  ggplot(aes(x = P_diff_y1, color = signal_fctr, fill = signal_fctr)) + 
  geom_density(alpha = 0.3) + 
  geom_rug()

train %>% 
  ggplot(aes(x = P_diff_x2, color = signal_fctr, fill = signal_fctr)) + 
  geom_density(alpha = 0.3) + 
  geom_rug()

train %>% 
  ggplot(aes(x = P_diff_y2, color = signal_fctr, fill = signal_fctr)) + 
  geom_density(alpha = 0.3) + 
  geom_rug()


train %>% summarise_signal()

train %>% group_by(Kplus_P > 5e4) %>% summarise_signal()


train %>% 
  ggplot(aes(x = Kplus_P, color = signal_fctr, fill = signal_fctr)) + 
  geom_density(alpha = 0.3) + 
  geom_rug()

summary(train)  

cor(train %>% select(where(is.numeric))) %>% View
train %>% select(where(is.numeric))
# train %>% select(starts_with("a"), signal, where(is.numeric))
train %>% select(starts_with("Kplus")) %>% head()
train %>% select(starts_with("pi")) %>% head()

corrplot::corrplot(cor(train %>% select_if(is.numeric)))

train %>% 
  group_by(B_PT < gamma_PT + Kplus_P + piminus_P) %>% 
  summarise_signal()

train %>% 
  group_by(B_PT < Kplus_P + piminus_P) %>% 
  summarise_signal()

train %>% 
  group_by(B_PT < gamma_PT) %>% 
  summarise_signal()

train %>% roc_auc(signal_fctr, gamma_PT)
train %>% roc_auc(signal_fctr, P_diff_sq)
train %>% roc_auc(signal_fctr, Kplus_P)
train %>% roc_auc(signal_fctr, Kplus_ETA)
train %>% roc_auc(signal_fctr, P_diff_x01)
train %>% roc_auc(signal_fctr, P_diff_x)
train %>% roc_auc(signal_fctr, P_diff_x1)
train %>% roc_auc(signal_fctr, P_diff_y)
train %>% roc_auc(signal_fctr, P_diff_y1)
train %>% roc_auc(signal_fctr, P_diff_y2)

