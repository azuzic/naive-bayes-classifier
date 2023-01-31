# "Naive Bayes SpamFilter"

Creating a naive Bayes spam filter in R. Project for Applied Statistics Course.

### Authors and assignments

-   Alesandro Žužić (azuzic@unipu.hr)
-   Luka Blašković (lblaskovi@unipu.hr)

### Documentation
- pdf file will go here

### Data
SMS Spam Collection Dataset
- The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.
- [It's available here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

### Short description of available functionalities
The naive Bayes is a classification algorithm that is suitable for binary and multiclass classification. It's based on applying Bayes' theorem. It is useful for making predictions and forecasting data based on historical results.
In this project, the classifier will be used to classify SMS messages into two labels: ham and spam, ham being legitimate message and spam being spam.
Classifier performance was messured using yardstick package. On a 5574 rows dataset, it resulted in high accuracy of 0.984 with Cohen's kappa value of 0.927, indicating almost perfect agreement.

Confusion matrix:
- #1202 messages were predicted to be ham, are indeed ham
- #5 messages predicted to be spam, are actually ham

- #21 messages predicted to be ham, were actually spam
- #166 messages predicted to be spam, were spam indeed

### Organization

[Juraj Dobrila University of Pula](http://www.unipu.hr/)  
[Pula Faculty of Informatics](https://fipu.unipu.hr/)  
Course: **Applied Statistics**, Academic Year 2022./2023.  
Mentor: **Siniša Sovilj** (https://fipu.unipu.hr/fipu/sinisa.sovilj, sinisa.sovilj@unipu.hr)
