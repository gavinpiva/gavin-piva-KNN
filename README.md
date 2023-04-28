# K-Nearest Neighbors (KNN) Model
## Introduction

This KNN model is designed to predict the income of individuals based on demographic data, such as age, education, and occupation, among others. The model uses a supervised learning approach to classify the individuals into two categories - "less than or equal to 50K" and "greater than 50K" based on the training data.

## Prerequisites

This KNN model requires the following dependencies:

numpy: For scientific computing with Python.

matplotlib: For creating static, animated, and interactive visualizations in Python.

pandas: For data manipulation and analysis.

argparse: For parsing command-line arguments.

## Preprocessing

Before training the model, the data is preprocessed by performing the following steps:

One-hot encoding: Categorical features are converted to numerical features using one-hot encoding.

Standardization: Continuous features are standardized.

## Model

The KNN model is a distance-based algorithm. For each test sample, the KNN algorithm calculates the distance to each training sample and selects the K nearest neighbors. The model then predicts the class of the test sample by taking the majority vote of the K neighbors.

## Usage

The KNN model can be run from the command line with the following arguments:

-k: The number of neighbors. Default value is 5.
-group: The name of the group to filter the testing data. Default value is None.
Examples: 
python KNN.py 

python KNN.py -k 5 -group sex_Male

python KNN.py -k 5 -group sex_Female

python KNN.py -k 5 -group race_White

python KNN.py -k 5 -group race_Black
## Outputs

The output of the KNN model is the accuracy of the prediction on the testing set. If the -group argument is provided, the accuracy will be calculated only on the samples that belong to that group.

If no -group argument is provided, the total number of sample size in training set will be displayed.

-------
Majority of this code was written in conjunction with Proffesor Jingsai Liang. 

This contains a KNN based model that uses demographics and general information of people from two csv files. 
Each csv file is a train and a test file. 

[data_test.csv](https://github.com/gavinpiva/gavin-piva-KNN/files/9023043/data_test.csv)
[data_train.csv](https://github.com/gavinpiva/gavin-piva-KNN/files/9023046/data_train.csv)

Based on the categorical variable you set this is able to determain a percentage of individuals in each group that make above or below 50 thousand dollars a year.
Here is what each result looks like:

![result](https://user-images.githubusercontent.com/65461919/176764967-852e760d-c02e-493c-af19-eebbb3bc0ab3.png)

## Things to keep in mind
What is the prediction of accuracy of your mode on male and female respectively?
Which group of people has higher prediction accuracy? Why?


The prediction accuracy for females has a higher prediction accuracy of 88% within the
group that makes less than 50K a year. This is because the data set has 91% of
females in this group. With this discrepancy in the data it is extremely likely that we can
accurately predict a female who makes less than 50K a year, but not accurately predict
the group in which females make more than 50K.


What is the prediction of accuracy of your mode on white and black respectively? Which
group of people has higher prediction accuracy? Why?


Similar to the prediction on the female group the black group has an 81% accuracy on
the group that is below 50K a year. This group has high accuracy because 88% of the
data is in this group. This means that a very low number of situations have this group
making over 50K a year. This difference in the data means that the prediction is much
higher in the higher percentage group.


Do you think the algorithmic decisions of your code convey any type of discrimination
against specific groups of population or minorities? If yes, where does the bias come
from?


The algorithmic decisions of this code is not what creates the discrimination in this
model. It is because of the discrepancy in the data. The higher percentage of certain
groups does not allow the model to correctly predict this group because there is an
imbalance in the 2 situations of the groups. The real world data is what creates the bias
of a group.


How would you mitigate the bias through modifying the data sets?

In our case this data is not balanced with each other. There are so many more cases of
one group than another. We could attempt to match the number of each group by
reducing the number of classifiers in the groups that already have a higher number of
cases. For example in the adults data set we use for this assignment there are only 111
out of 1000 cases of individuals who are black around 10%. We could attempt to reduce
the white category to even out our data and potentially reduce bias. The data that we
use has primarily white individuals this causes bias towards predicting one group


If you have any questions about this code feel free to ask!
