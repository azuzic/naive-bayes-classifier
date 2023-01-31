#install.packages(c("dplyr", "stringr", "rsample"))
#install.packages("rsample")
library("dplyr")
library("stringr")

#Read raw data from spam.csv
#Data source: SMS Spam Collection Dataset - https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
set.seed(0303088177)

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
train_sample <- rsample::training(split)
test_sample <- rsample::testing(split)

library(ggplot2)

#Data frame with the dimensions of each dataset
data_dim <- data.frame(Data = c("Train", "Test"),
                       Dimensions = c(dim(train_sample)[1], dim(test_sample)[1]))

#Bar plot of the ratio of train and test datasets
ggplot(data_dim, aes(x = Data, y = Dimensions)) +
  geom_col() +
  ggtitle("Ratio between Train and Test Data") +
  xlab("Data Set") +
  ylab("Number of Observations")

# distributions of the categories (Ham ~85% | Spam ~15%) in train data
prop.table(table(train_sample$label))

# distributions of the categories (Ham ~85% | Spam ~15%) in test data
prop.table(table(test_sample$label))


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

train_sample <- train_sample %>%
  mutate(msg_list = string_cleaner(.$msg))

#Data examples
head(train_sample)

#1. label:ham, msg: "Fair enough, anything going on?", msg_list: c("fair", "enough", "anything", "going", "on")
#2. label:ham, msg: "	Hi :)finally i completed the course:)", msg_list: c("hi", "finally", "completed", "the", "course")
#3. label:spam, msg: "07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile", msg_list:c("_longnum_", "rodger", "burns", "msg", "we", "tr [...]


#Building the vocabulary
#Next step is to calculate PROBABILITIES of any give nword appearing in each type of message
#Extracting all unique words in the dataset. 
vocab <- train_sample %>%
  select(msg_list) %>%
  unlist() %>%
  unique() %>%
  tibble::enframe(name = NULL, value = "word")

vocab

# A tibble: 7,837 × 1
#word   
#<chr>  
#1 go       
#2 until    
#3 jurong   
#4 point    
#5 crazy    
#6 available
#7 only     
#8 in       
#9 bugis    
#10 great 
# … with 7,827 more rows

#Same thing as previous function, but for ham words, non-uniques included
ham_vocab <- train_sample %>%
  filter(label == "ham") %>%
  select(msg_list) %>%
  tibble::deframe() %>%
  unlist()
head(ham_vocab)

#Same thing as previous function, but for spam words, non-uniques included
spam_vocab <- train_sample %>%
  filter(label == "spam") %>%
  select(msg_list) %>%
  tibble::deframe() %>%
  unlist()
head(spam_vocab)

#Count how often the words appear in each category and join with ham_vocabs, spam_vocabs
vocab <- table(ham_vocab) %>%
  tibble::as_tibble() %>%
  rename(ham_n = n) %>%
  left_join(vocab, ., by = c("word" = "ham_vocab"))

vocab <- table(spam_vocab) %>%
  tibble::as_tibble() %>%
  rename(spam_n = n) %>%
  left_join(vocab, ., by = c("word" = "spam_vocab"))

vocab

# A tibble: 7,922 × 3
#word      ham_n spam_n
#<chr>     <int>  <int>
#1 go          191     22
#2 until        16      5
#3 jurong        1     NA
#4 point        12     NA
#5 crazy         7      4
#6 available    13      1
#7 only        105     55
#8 in          617     64
#9 bugis         5     NA
#10 great        74      8
# … with 7,912 more rows


#Next step is to turn these counts in probability since that is what we need for classification.

word_n <- c("unique" = nrow(vocab),
            "ham" = length(ham_vocab),
            "spam" = length(spam_vocab))
class_probs <- prop.table(table(train_sample$label))
class_probs #Ham~0.866, Spam~0.134

library(ggplot2)

#Creating a dataframe with data probabilities that it is Ham or Spam
class_probs_df <- data.frame(Class = c("Ham", "Spam"),
                             Probability = c(class_probs["ham"], class_probs["spam"]))

#Bar plot of probability ratios (Ham data/Spam data)
ggplot(class_probs_df, aes(x = Class, y = Probability)) +
  geom_col() +
  ggtitle("Class Probabilities") +
  xlab("Class") +
  ylab("Probability")

#Helper function for calculating the probability of word appearing in a given category (Ham,Spam)
#This functions relates the amount of words in the category to the amount of total words in the category and adds Laplacian smoothing to ensure the result is never 0.
word_probabilities <- function(word_n, category_n, vocab_n, smooth = 1) {
  prob <- (word_n + smooth) / (category_n + smooth * vocab_n)
  prob
}
#The function takes the frequency of a word, the amount of words in the category, the amount of total unique words and a smoothing value 
#It returns the probability of the word belonging to a category
#Also, replace NA counts with 0 and apply word_probabilities helper function to each row
vocab <- vocab %>%
  tidyr::replace_na(list(ham_n = 0, spam_n = 0)) %>%
  rowwise() %>%
  mutate(ham_prob = word_probabilities(
    ham_n, word_n["ham"], word_n["unique"])) %>%
  mutate(spam_prob = word_probabilities(
    spam_n, word_n["spam"], word_n["unique"])) %>%
  ungroup()

#Final vocab (data frame of possibilities)
vocab

# A tibble: 7,922 × 5
#word      ham_n spam_n  ham_prob spam_prob
#<chr>     <int>  <int>     <dbl>     <dbl>
#1 go          191     22 0.00351   0.00115  
#2 until        16      5 0.000311  0.000299 
#3 jurong        1      0 0.0000365 0.0000498
#4 point        12      0 0.000237  0.0000498
#5 crazy         7      4 0.000146  0.000249 
#6 available    13      1 0.000256  0.0000997
#7 only        105     55 0.00194   0.00279  
#8 in          617     64 0.0113    0.00324  
#9 bugis         5      0 0.000110  0.0000498
#10 great        74      8 0.00137   0.000449 
# … with 7,912 more rows

#Classification
#By multiplying the probabilities for all words in the message given each category 
#and also adding the baseline probability of the categories into the product.


#Takes raw message as input and returns a classification
classifier <- function(msg, prob_df, ham_p = 0.5, spam_p = 0.5) {
  clean_message <- string_cleaner(msg) %>% unlist()
  
  probs <- sapply(clean_message, function(x) {
    filter(prob_df, word == x) %>%
      select(ham_prob, spam_prob)
  })
  
  if (!is.null(dim(probs))) {
    ham_prob <- prod(unlist(as.numeric(probs[1, ])), na.rm = TRUE)
    spam_prob <- prod(unlist(as.numeric(probs[2, ])), na.rm = TRUE)
    ham_prob <- ham_p * ham_prob
    spam_prob <- spam_p * spam_prob
    
    if (ham_prob > spam_prob) {
      classification <- "ham"
    } else if (ham_prob < spam_prob) {
      classification <- "spam"
    } else {
      classification <- "unknown"
    }
  } else {
    classification <- "unknown"
  }
  
  classification
}

#Applying to the test data set
#The function takes 4 inputs: the message, a data frame of probabilites (vocab) and baseline proabilities (class_probs["ham"] and class_probs["spam"])
#The message wil be our test sample
#The function call takes 3-5 minutes on somewhat decent machine
spam_classification <- sapply(test_sample$msg,
                              function(x) classifier(x, vocab, class_probs["ham"],
                                                     class_probs["spam"]), USE.NAMES = FALSE)

#Let's see how it's performing
fct_levels <- c("ham", "spam", "unknown")

test_sample <- test_sample %>%
  mutate(label = factor(.$label, levels = fct_levels),
         .pred = factor(spam_classification, levels = fct_levels))

performance <- yardstick::metrics(test_sample, label, .pred)

performance

# A tibble: 2 × 3
#.metric  .estimator .estimate
#<chr>    <chr>          <dbl>
#1 accuracy multiclass     0.984
#2 kap      multiclass     0.927

#On a 5574 rows dataset, it resulted in high accuracy of 0.984 with Cohen's kappa value of 0.927, indicating almost perfect agreement.

library(ggplot2)
library(reshape2)

#Confusion matrix
cm <- table(paste("actual", test_sample$label), paste("pred", test_sample$.pred))
#
#            pred ham pred spam
#actual ham      1202         5
#actual spam       21       166

cm_melted <- melt(cm)
ggplot(cm_melted, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile() + 
  geom_text(aes(label=value), color="black", size=3.5) +
  scale_fill_gradient(low = "white", high = "steelblue") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Confusion Matrix", x = "Prediction", y = "Actual")

#1202 messages were predicted to be ham, and are indeed ham
#5 messages predicted to be spam, are actually ham

#21 messages predicted to be ham, were actually spam
#166 messages predicted to be spam, were spam indeed
