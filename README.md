## NNregressionNFL2019
Simple NN regression example using TensorFlow 2.0 and Sequential()

This is an example ANN regression using TensorFlow and the Sequential() model. 
The import stuff loads mystic, but that package is not used (yet).

There are a few steps to running the net:

1. clone the repository, set path to repo location:
  path='C:/Users/Computer/Documents/NFL' ... actually path='your path to repo here'
2. Lines 89-82 set 

   **train_size = 0.67  # what fraction of the data for training**  
   **epochs = 100  # iterations of estimate**  
   **randomSeed= int(7)**
   
   
The data set, NFL2019.csv is a collection of game stats for the 32 teams of the NFL.  The idea is to take a subset of 'train_size' to train the network, and evaluate on the remaing data.  Change the train_size (between 0 and 1) to train/test different proportions.

Epochs is the number of steps in the forward/backward estimation process. Too few and you underfit the model; too many and you overfit.  Try it!

randomSeed is set to 7 to basicallyt replicate results using diffeent 'tuning' values

The output is the modeled wins/losses for each team, along with the actual wins/losses.  Use different random seeds to see how poor this model is-- not enough data.  Also this is not a prediction model.  Train data were used to fit the model; the test data were used to evaluate the fit.   If one really wanted to see how well the model works, train/test on 2018 data and then predict results for 2019 season.   I don't recommend that one uses this model for anything else but learning to use Sequential() from TensorFlow.


