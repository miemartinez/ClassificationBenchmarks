# Classification benchmarks
**This project was developed as part of the spring 2021 elective course Cultural Data Science - Visual Analytics at Aarhus University.**

__Task:__ The task for this project is to build two classification models i.e., a logistic regression and a neural network. 
These models are created as command-line tools that perform a simple classification task on the full MNIST data and provide benchmark scores for easily evaluating their performance. <br>

The repository contains two scripts in the src folder. Both scripts can be run without inputs (as they have defaults) or the user can specify these. <br>
The first script (lr_mnist.py) is used to train a logistic regression classifier on the MNIST data using Scikit-Learn. Here, the user can experiment with different solvers and penalties to see whether this could improve the model. These parameters can be specified in the command line. <br> 
The second script (nn_mnist.py) uses Scikit-Learn again to build and train a neural network classifier. The model architecture depends on how many layers are specified in the command line.<br> 

The output of the python script is also provided in the out folder. This contains the benchmarks for both models in the form of a classification report saved as csv and a confusion matrix for the logistic regression model saved as png.

Additionally, both scripts allow the user to test the model predictions by specifying the path to a test image (as the one provided in the data folder) and getting the predicted value for which number it represents.

__Dependencies:__ <br>
To ensure dependencies are in accordance with the ones used for the scripts, you can create the virtual environment ‘classifier_environment"’ from the command line by executing the bash script ‘create_classifier_venv.sh’. 
```
    $ bash ./create_classifier_venv.sh
```
This will install an interactive command-line terminal for Python and Jupyter as well as all packages specified in the ‘requirements.txt’ in a virtual environment. 
After creating the environment, it will have to be activated before running the classifier scripts.
```    
    $ source classifier_environment/bin/activate
```
After running these two lines of code, the user can commence running one of the scripts. <br>

### How to run lr-mnist.py <br>
The script lr-mnist.py can run from command line without additional input. 
However, the user can experiment with the model by specifying train/test split size, tolerance level, penalty method, solver method and maximum iteration in the command line argument. 
Additionally, the filename for the classification report and the path to an unseen image can defined. 
The outputs of the script are a csv file of the model classification, a png file of the confusion matrix and (if provided with unseen image) a printed prediction in the terminal.
**Before defining the solver and penalty the user should note that there are some conflicts.** 
In the Sci-Kit Learn documentation it is stated that multiclass problems can only be dealt with by using the solvers ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’. 
Therefore, only one of these four solvers should be chosen in the command line. If none is chosen, the default is ‘saga’. 
Furthermore, all solvers can handle ‘l2’ regularization or no penalty. However, only saga can also handle ‘l1’ and ‘elasticnet’ penalty.

__Parameters:__ <br>
```
    train_size: float <number-between-0-and-1>, default = 0.8
    classification_tolerance: float <number-between-0-and-1>, default = 0.1
    penalty: str <regularization-penalty>, default = ‘none’
    solver: str <solver-for-optimization>, default = ‘saga’
    max_iter: int <max-iterations>, default = 100
    filename: str <choose-filename-for-csv>, default = ‘lr_classification_metrics.csv’ 
    path2image: str <path-to-unseen-image>

```
    
__Usage:__ <br>
```
    lr-mnist.py -t <train-size> -c <classification-tolerance> -pe <penalty-method> -s <solver-method> -m <max-iterations> -f <chosen-filename> -p <path-to-image>
```
    
__Example:__ <br>
```
    $ cd src
    $ python3 lr-mnist.py -t 0.7 -c 0.05 -pe l2 -s saga -p ../data/test.png

```


### How to run nn-mnist.py <br>

The script nn-mnist.py can run from command line without additional input. 
However, to experiment with the model, the user can specify train/test split size, tolerance level, penalty method, solver method and maximum iteration argument. 
Furthermore, the user can define the filename for the classification report and the path to an unseen image. 
The outputs of the script are a csv file of the model classification and a printed prediction in the terminal if given an unseen image in the command line.


__Parameters:__ <br>
```
    train_size: float <number-between-0-and-1>, default = 0.8
    filename: str <choose-filename-for-csv>, default = "nn_classification_metrics.csv"
    hidden_layer1: int <nodes-in-hidden-layer-1>, default = 32
    hidden_layer2: int <nodes-in-hidden-layer-2>, default = 0
    hidden_layer3: int <nodes-in-hidden-layer-3>, default = 0
    hidden_layer4: int <nodes-in-hidden-layer-4>, default = 0
    epochs: int <number-of-epochs>, default = 1000
    path2image: str <path-to-unseen-image>

```
    
__Usage:__ <br>
```
    nn-mnist.py -t <train-size> -f <chosen-filename> -hl1 <hidden-layer1> -hl2 <hidden-layer2> -hl3 <hidden-layer3> -hl4 <hidden-layer4> -e <epochs> -p <path-to-image>
```
    
__Example:__ <br>
```
    $ cd src
    $ python3 nn-mnist.py -t 0.7 -f "nn_evaluation.csv" -hl1 64 -hl2 32 -hl3 16 -e 1000 -p ../data/test.png

```

The code has been developed in Jupyter Notebook and tested in the terminal on Jupyter Hub on worker02. I therefore recommend cloning the Github repository to worker02 and running the scripts from there. 

### Results:
With the logistic regression classifier, the best obtained result was a weighted average accuracy of 92%. 
However, this was the result obtained across almost all combinations of solvers and penalty methods 
(with the exception of using the newton-cg solver with no penalty which converged after increasing max iterations to 500 and had an accuracy of 91%). 
The neural network model had a weighted average accuracy of 97% when run for 1000 epochs with an architecture of 4 hidden layers [784, 64, 32, 16, 10]. 
So, this is a decent increase from the logistic regression classifier. However, they both make the correct prediction on the test image. 



