#!/usr/bin/env python
"""
Training a neural network classifier for the MNIST data and evaluates the model. Saving the classification metrics from evaluation as csv file in the out folder as well as printing to the terminal. Can also be used for classifying unseen image of a number between 0-9 by specifying path to the png image. 

Parameters:
    train_size: float <number-between-0-and-1>, default = 0.8
    filename: str <choose-filename-for-csv>, default = "nn_classification_metrics.csv"
    hidden_layer1: int <nodes-in-hidden-layer-1>, default = 32
    hidden_layer2: int <nodes-in-hidden-layer-2>, default = 0
    hidden_layer3: int <nodes-in-hidden-layer-3>, default = 0
    hidden_layer4: int <nodes-in-hidden-layer-4>, default = 0
    epochs: int <number-of-epochs>, default = 1000
    path2image: str <path-to-unseen-image>
Usage:
    nn_mnist.py -t <train-size> -f <chosen-filename> -hl1 <hidden-layer1> -hl2 <hidden-layer2> -hl3 <hidden-layer3> -hl4 <hidden-layer4> -e <epochs> -p <path-to-image>
Example:
    $ python3 nn-mnist.py -t 0.8 -f "nn_evaluation.csv" -hl1 64 -hl2 32 -hl3 16 -e 1000 -p ../data/test.png
    
## Task
- Making a neural network classifier as a command-line tools which can be used to perform a simple classification task on the MNIST data and print the output to the terminal.
- Have the script save the classifier report in a folder called out, as well as printing it to screen. 
- The user should be able to define the filename as a command line argument
- Allow the user to define train size using command line arguments 
- The user can choose the number of hidden layers (up to three) and the number of nodes in each layer.
"""
# import libraries
import os
import sys
sys.path.append(os.path.join(".."))
import argparse

# import teaching utils
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

# Neural networks with numpy
from utils.neuralnetwork import NeuralNetwork

# machine learning tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets


# argparse 
ap = argparse.ArgumentParser()
# adding argument
ap.add_argument("-t", "--train_size", 
                default = 0.8, 
                help = "Percentage of data to use for training")

ap.add_argument("-f", "--filename",
                default = "nn_classification_metrics.csv", 
                help = "Define filename for csv of classification metrics")

ap.add_argument("-hl1", "--hidden_layer1",
                default = 32, 
                type=int,
                help = "Number of nodes for the first hidden layer")

ap.add_argument("-hl2", "--hidden_layer2",
                default = 0, 
                type = int,
                help = "Number of nodes for the second hidden layer")

ap.add_argument("-hl3", "--hidden_layer3",
                default = 0, 
                type = int,
                help = "Number of nodes for the third hidden layer")

ap.add_argument("-hl4", "--hidden_layer4",
                default = 0, 
                type = int,
                help = "Number of nodes for the fourth hidden layer")

ap.add_argument("-e", "--epochs",
                default = 1000,
                type = int,
                help = "Number of epochs to train over")

ap.add_argument("-p", "--path2image",
                required = False,
                help = "If you want to test the model on unseen data, add the file path of an unseen image")
                

# parsing arguments
args = vars(ap.parse_args())




def main(args):
    '''
    Main function that should be executed when running from the command line.
    '''
    # Getting the arguments for the file and model training parameters 
    filename = args["filename"]
    train_size = float(args["train_size"])
    hidden_layer1 = args["hidden_layer1"]
    hidden_layer2 = args["hidden_layer2"]
    hidden_layer3 = args["hidden_layer3"]
    hidden_layer4 = args["hidden_layer4"]
    epochs = args["epochs"]
    # path to unseen image if path is specified
    path2image = 0
    if args["path2image"] is not None:
        path2image = args["path2image"]
    # otherwise set image path to False
    else:
        path2image == False
    
    # Create class object with train size and tolerance level
    nn_classifier = NeuralNetClassifier(train_size, 
                                        filename, 
                                        hidden_layer1, 
                                        hidden_layer2, 
                                        hidden_layer3, 
                                        hidden_layer4, 
                                        epochs, 
                                        path2image)
    
    # use method train_class
    # training model and saving it (also returning the test data)
    nn_model, X_test_scaled, y_test = nn_classifier.train_class()
    
    
    # use method eval_classifier
    nn_classifier.eval_classifier(nn_model, 
                                  X_test_scaled, 
                                  y_test)
    
    # test classifier if an unseen image is passed as argument. Otherwise pass.
    if path2image == False:
        pass
    else:
        nn_classifier.test_model(nn_model)


    
class NeuralNetClassifier:
    
    def __init__(self, train_size, filename, hidden_layer1, hidden_layer2, hidden_layer3, hidden_layer4, epochs, path2image):
        '''
        Constructing the Classification object
        '''
        # train size
        self.train_size = train_size
        # user chosen filename
        self.filename = filename
        
        # hidden layers
        self.hidden_layer1 = hidden_layer1
        self.hidden_layer2 = hidden_layer2
        self.hidden_layer3 = hidden_layer3
        self.hidden_layer4 = hidden_layer4
        
        self.epochs = epochs
        
        # unseen image
        self.path2image = path2image
        
        # fetch the mnist data
        print("\n-- Fetching the data --")
        X, y = fetch_openml("mnist_784", version = 1, return_X_y=True)
        # make sure the data is type numpy array 
        self.X = np.array(X)
        self.y = np.array(y)          
    
    
    
    def train_class(self):
        '''
        Creating training data and training the classification model
        '''
        # Create training and test data
        X_train, X_test, y_train, y_test = train_test_split(self.X, 
                                                            self.y,
                                                            random_state=9,
                                                            train_size=self.train_size) # argparse - make this the default
        # Min-Max scaling
        X_train_scaled = (X_train - X_train.min())/(X_train.max() - X_train.min())
        X_test_scaled = (X_test - X_test.min())/(X_test.max() - X_test.min())
        
        # convert labels from integers to vectors
        y_train = LabelBinarizer().fit_transform(y_train)
        y_test = LabelBinarizer().fit_transform(y_test)

        # train network using the NeuralNetwork class from utils
        print("\n[INFO] training network...")
        
        # if only one hidden layer is defined
        if self.hidden_layer1 > 0 and self.hidden_layer2 == 0 and self.hidden_layer3 == 0 and self.hidden_layer4 == 0:
            print(f"\nYour model is using one hidden layer with the size [{self.hidden_layer1}]")
            # train the model with one hidden layer
            nn_model = NeuralNetwork([X_train_scaled.shape[1], 
                                      self.hidden_layer1, 
                                      10])
            
        # else if two hidden layers are defined
        elif self.hidden_layer1 > 0 and self.hidden_layer2 > 0 and self.hidden_layer3 == 0 and self.hidden_layer4 == 0:
            print(f"\nYour model is using two hidden layers with the size [{self.hidden_layer1}, {self.hidden_layer2}]")
            # train the model with two hidden layers
            nn_model = NeuralNetwork([X_train_scaled.shape[1], 
                                      self.hidden_layer1, 
                                      self.hidden_layer2, 
                                      10])
            
        # else if three hidden layers are defined
        elif self.hidden_layer1 > 0 and self.hidden_layer2 > 0 and self.hidden_layer3 > 0 and self.hidden_layer4 == 0:
            print(f"\nYour model is using three hidden layers with the size [{self.hidden_layer1}, {self.hidden_layer2}, {self.hidden_layer3}]")
            # train the model with three hidden layers 
            nn_model = NeuralNetwork([X_train_scaled.shape[1], 
                                      self.hidden_layer1, 
                                      self.hidden_layer2, 
                                      self.hidden_layer3, 
                                      10])
            
        # else if four hidden layers are defined
        elif self.hidden_layer1 > 0 and self.hidden_layer2 > 0 and self.hidden_layer3 > 0 and self.hidden_layer4 > 0:
            print(f"\nYour model is using four hidden layers with the size [{self.hidden_layer1}, {self.hidden_layer2}, {self.hidden_layer3}, {self.hidden_layer4}]")
            # train the model with four hidden layers
            nn_model = NeuralNetwork([X_train_scaled.shape[1], 
                                      self.hidden_layer1, 
                                      self.hidden_layer2, 
                                      self.hidden_layer3, 
                                      self.hidden_layer4, 
                                      10])
            
        # printing progress
        print("\n[INFO] {}".format(nn_model))
        nn_model.fit(X_train_scaled, y_train, epochs=self.epochs)
        
        return nn_model, X_test_scaled, y_test
    
    
    
    def eval_classifier(self, nn_model, X_test_scaled, y_test):
        # Create out directory if it doesn't exist in the data folder
        dirName = os.path.join("..", "out")
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("\nDirectory " , dirName ,  " Created ")
        else:   
            print("\nDirectory " , dirName ,  " already exists")
            
        # evaluate network
        print(["\n[INFO] evaluating network..."])
        # testing the model on the test data
        predictions = nn_model.predict(X_test_scaled)
        predictions = predictions.argmax(axis=1)
        # calculating classification metrics
        classification_metrics = classification_report(y_test.argmax(axis=1), predictions) # return as dictionary
        print(classification_metrics) # Print in terminal
        
        # define as pandas dataframe and save as csv in the out folder
        path = os.path.join("..", "out", self.filename)
        # transpose and make into a dataframe
        classification_metrics_df = pd.DataFrame(classification_report(y_test.argmax(axis=1), predictions, output_dict = True)).transpose()
        # saving as csv
        classification_metrics_df.to_csv(path)
        # print that the csv file has been saved
        print(f"\nClassification metrics are saved as {path}")
        
        
        
    def test_model(self, nn_model):
        '''
        Testing the neural network model on an unseen image.
        Saving the model probabilities for the different digits.
        Printing the label with the highest probability.
        '''
        print("\n--Testing the image you gave me--")
        # reading image from path
        test_image = cv2.imread(self.path2image)
        # gray scaling image
        gray = cv2.bitwise_not(cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY))
        # compressing image to match the classifier
        compressed = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        # defining classes
        classes = sorted(set(self.y))
        
        # Reshape array
        test_probs = nn_model.predict(compressed.reshape(1,784))
        # plot prediction
        plot_path = os.path.join("..", "out", "label_predictions.png")
        sns_plot = sns.barplot(x=classes, y=test_probs.squeeze());
        plt.ylabel("Probability");
        plt.xlabel("Class")
        fig = sns_plot.get_figure()
        fig.savefig(plot_path)
        # print that the figure has been saved
        print(f"\nThe label probability plot is saved as {plot_path}")
        
        # find and save the label with highest probability
        idx_cls = np.argmax(test_probs)
        # print predictied label
        print(f"I think that this is class {classes[idx_cls]}")
       
    
if __name__ == "__main__":
    main(args)

