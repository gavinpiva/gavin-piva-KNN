# gavin-piva-KNN

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def one_hot(data, column):
    data = pd.get_dummies(data, columns = [column])
    return data
    
def standardize(data, column):
    X = data[column]
    data[column] = (X - np.mean(X))/ np.std(X)
    return data

def pre_processing(data):
    category_list = ['workclass','education','marital-status','occupation','relationship',
                    'race','sex','native-country']
    continuous_list = ['age', 'education-num', 'hours-per-week']
    
    for column in category_list:
        data = one_hot(data, column)
    
    for column in continuous_list:
        data = standardize(data, column)
        
    return data    
#The code above was given, the KNN is original
class KNN:
    def __init__(self, k=5, group=None):
        self.k = k
        self.group = group 
        # Read the data
        data_train = pd.read_csv('data_train.csv')
        data_test = pd.read_csv('data_test.csv')

        # Pre-process the data 
        X_train = data_train.drop('salary',axis=1)
        self.X_train = pre_processing(X_train)
        self.y_train = data_train['salary']

        X_test = data_test.drop('salary',axis=1)
        self.X_test = pre_processing(X_test) 
        self.y_test = data_test['salary']

    def get_test_data(self):
        # Filter the testing data based on group 
        if self.group:
            mask = self.X_test[self.group]==1
            self.X_test = self.X_test[mask]
            self.y_test = self.y_test[mask]

        self.X_train, self.y_train, self.X_test, self.y_test = self.X_train.values, self.y_train.values, self.X_test.values, self.y_test.values

    def get_train_data_size(self):
        if self.group:
            mask = self.X_train[self.group]==1
            less50 = self.y_train[mask] == '<=50K'
            print("The percentage of people whose salary is less than 50K in the group of "+self.group+" in training set is", sum(less50)/sum(mask))
        else:
            print("The total number of sample size in training set is",len(self.X_train))

    def distance(self, p1, p2):
        #Distance finction to find the distance between two specific points
        dist = np.sqrt(np.sum(np.square(p1 - p2)))
        return dist

    def run(self): 
        #KNN Model based function
        accuracy = 0

        # Your code goes here:
        correct = 0
        for p in range(len(self.X_test)):
            pqArr = []
            for q in self.X_train:
                pqDist = self.distance(self.X_test[p],q)
                pqArr.append(pqDist)
            mask = np.argsort(pqArr)
            trainer = self.y_train[mask[:self.k]]
            great = 0
            less = 0
            for i in trainer:
                if i == ">50K":
                    great +=1
                else:
                    less +=1
            if great > less:
                if self.y_test[p] == ">50K":
                    correct+=1
            else:
                if self.y_test[p] == "<=50K":
                    correct+=1

        predicted = correct / len(self.y_test)
        accuracy = predicted

        if self.group:
            print('The accuracy of this model on the testing set in the group of', self.group, 'is ', accuracy)
        else:
            print('The accuracy of this model on the testing set is ', accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN')
    parser.add_argument('-k', dest='k', required = False, default=5, type = int, help='number of neighbors')
    parser.add_argument('-group', dest='group', required = False, default=None, type = str, help='name of group')

    args = parser.parse_args()

    obj = KNN(args.k, args.group)
    obj.get_train_data_size()
    obj.get_test_data()
    obj.run()
    
    #run functions for inline commands
    # python KNN.py 
    # python KNN.py -k 5 -group sex_Male
    # python KNN.py -k 5 -group sex_Female
    # python KNN.py -k 5 -group race_White
    # python KNN.py -k 5 -group race_Black
