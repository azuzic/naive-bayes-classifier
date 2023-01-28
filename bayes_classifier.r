#install.packages(c("dplyr", "stringr", "rsample"))
#install.packages("rsample")
library("dplyr")
library("stringr")

spam <- read.csv("spam.csv")


spam <- spam %>%
  tidyr::unite(col = "msg", 2:5, sep = " ", na.rm = TRUE) %>% 
  rename("label" = v1)

spam

library("rsample")
split <- rsample::initial_split(spam, strata = label)
train_spam <- rsample::training(split)
test_spam <- rsample::testing(split)