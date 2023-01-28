#install.packages(c("dplyr", "stringr", "rsample"))
#install.packages("rsample")
library("dplyr")
library("stringr")

#Read raw data from spam.csv
#Data source: SMS Spam Collection Dataset - https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
spam <- read.csv("spam.csv")

#Uniting multiple columns into one by pasting strings together
#This merges the last columns into a single one and renames first one to 'label'
spam <- spam %>%
  tidyr::unite(col = "msg", 2:5, sep = " ", na.rm = TRUE) %>% 
  rename("label" = v1)

#label (Ham - 'Good SMS', Spam - 'Bad SMS')
#msg (sms message content)
head(spam)

#Splitting data into train and test samples in order not to over sample or under sample the amount of spam in each sample.
library("rsample")
#Variable strata is used to conduct stratified sampling
split <- rsample::initial_split(spam, strata = label)
train_spam <- rsample::training(split)
test_spam <- rsample::testing(split)

# distributions of the categories (Ham ~85% | Sam ~15%) in train data
prop.table(table(train_spam$label))

# distributions of the categories (Ham ~85% | Sam ~15%) in test data
prop.table(table(test_spam$label))

# To process the messages, we have to split them first into individual words and clean the data. 
string_cleaner <- function(text_vector) {
  tx <- text_vector %>%
    str_replace_all("[^[:alnum:] ]+", "") %>% #Remove all punctuation
    str_to_lower() %>% #Make everything lower case
    str_replace_all("\\b(http|www.+)\\b", "_url_") %>% #Transform everything that looks like hyperlink to '_url_'
    str_replace_all("\\b(\\d{7,})\\b", "_longnum_") %>% #Transform long sequences of numbers to '_longnum_'
    str_split(" ") #Split on spaces
  
  tx <- lapply(tx, function(x) x[nchar(x) > 1]) #Remove 1 char words
  
  tx
}

train_spam <- train_spam %>%
  mutate(msg_list = string_cleaner(.$msg))

#Data examples
head(train_spam)

#1. label:ham, msg: "Fair enough, anything going on?", msg_list: c("fair", "enough", "anything", "going", "on")
#2. label:ham, msg: "	Hi :)finally i completed the course:)", msg_list: c("hi", "finally", "completed", "the", "course")
#3. label:spam, msg: "07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile", msg_list:c("_longnum_", "rodger", "burns", "msg", "we", "tr [...]


