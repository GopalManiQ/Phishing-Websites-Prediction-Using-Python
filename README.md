# Phishing-Websites-Prediction
Developed a model that predicts/classifies whether a website is a Phishing website.
# Dataset
The dataset is downloaded from UCI machine learning repository. The dataset contains 31 columns, with 30 features and 1 target. The dataset has 2456 observations.
# PreProcessing
## Normalization
In order to normalize the range of the feature values over all samples, we standardized(aka normlize) the continuous feature values according to the formula: x(max−min)+min

x represents the column vector containing continuous feature values of each of the continuous-valued attributes for all samples.
# Model Selection
To Fit the models we split the dataset in to training annd testing data and our model will be trained on training data.
The choosen classifiers are:
1. Logistic Regression
2. Support Vector Machines
3. Decision Tree
4. Random Forest

## Ensemble methods
The choosen classifiers are:
1. Ada Boost
2. Gradient Boosting
3. Max Voting

# Modelling and Results
