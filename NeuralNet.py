import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import Utility

np.random.seed(42) # for reproductibility

class NeuralNet:
    def __init__(self, hidden_layer_sizes=(4,), activation='identity', error_cost='mse', batch_size=1,
                 learning_rate=0.1, eta_decay=False, warm_start=False, 
                 early_stopping=False, patience=0, epoch=200):
      self.n_layers  = len(hidden_layer_sizes)
      self.n_inputs  = 0
      self.n_outputs = 0
      self.n_epoch   = epoch
      self.batch_size  = batch_size
      self.early_stopping = early_stopping
      self.patience       = patience
      self.learning_rate = learning_rate
      self.eta_decay     = eta_decay
      self.warm_start    = warm_start
      self.has_trained   = False
      
      self.weights = [None] * (self.n_layers + 1) # +1: output layer
      self.biases  = [None] * (self.n_layers + 1)
      self.Z       = [None] * (self.n_layers + 1)
      self.A       = [None] * (self.n_layers + 1)
      self.df      = [None] * (self.n_layers + 1)
      
      self.hidden_layer_sizes = hidden_layer_sizes # To later determine
                                                    # weights matrices dim.
      
      if   activation == 'tanh'    : self.activation = Utility.tanh
      elif activation == 'sigmoid' : self.activation = Utility.sigmoid
      elif activation == 'relu'    : self.activation = Utility.relu
      elif activation == 'sintr'   : self.activation = Utility.sintr 
      else : self.activation = Utility.identity


      if    error_cost == 'cross-entropy' : self.error_cost = Utility.cross_entropy_cost
      else: self.error_cost = Utility.MSE_cost
          
      return None
    
    def __weights_initialization(self, X, y):
      # Initialize hidden layers
      n_cols  = X.shape[1] # nb of features of X
      
      self.n_inputs = X.shape[1]
      for l in range (0, self.n_layers):
          n_lines         = self.hidden_layer_sizes[l]
          self.weights[l] = np.random.uniform(-1, 1, (n_lines,n_cols))
          self.biases[l]  = np.random.uniform(-1, 1, (n_lines, 1)) # bias: vector
          n_cols          = n_lines
      
      # Initialize output layer
      l_out = self.n_layers # index of the last layer
      self.n_outputs = y.shape[1]
      self.weights[l_out] = np.random.uniform(-1, 1, (self.n_outputs,n_cols) )
      self.biases[l_out]  = np.random.uniform(-1, 1, (self.n_outputs, 1) )
      
      return None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, val=0.2):
      if not self.has_trained:
        self.__weights_initialization(X_train, y_train)
      
      if X_val is None:
        X_train, X_val, y_train, y_val = \
          train_test_split(X_train, y_train, test_size=val, random_state=42)
      
      epoch_train_error = []
      epoch_val_error  = []
      
      for e in range(0, self.n_epoch):
          ave_train_error = 0
          batch_train_error = []
          processed_batch = 0
          
          # Re-sample the training data
          X_train, y_train = shuffle(X_train, y_train) # sklearn
          
          # Training / Go over the data in batches
          for start in range(0, X_train.shape[0], self.batch_size):
              batch_X_train = X_train[start:start+self.batch_size].transpose()
              batch_y_train = y_train[start:start+self.batch_size].transpose()
              
              # Forward pass
              batch_train_error.append(self.__feed_forward(batch_X_train, batch_y_train)[0])
              ave_train_error += batch_train_error[processed_batch]
              
              # Backpropagation
              self.__backward_pass(batch_X_train, batch_y_train, e, True)
              
              processed_batch += 1
              # #### Training / End batch process for current epoch

          # Training error
          ave_train_error /= len(batch_train_error)
          epoch_train_error.append(ave_train_error)
              
          # Validation error
          val_error = self.__feed_forward(X_val.transpose(), y_val.transpose())[0]
          epoch_val_error.append(val_error)
          
          if e % 10 == 0:
              print("* Epoch " + str(e) + " -- Error :   Train : " + \
                    "{:.4f}".format(ave_train_error) + "    Validation : "+ \
                        "{:.4f}".format(val_error))
          
          #### END OF CURRENT EPOCH ####
      #### END OF EPOCHS ####
      
      self.has_trained = True
      
      # Plot error as a function of training epoch
      fig = plt.figure(figsize=(12,8))
      fig.suptitle('Evolution of error during training', fontsize=20)
      plt.plot(epoch_train_error, label="Train")
      plt.plot(epoch_val_error,  label="Validation")
      plt.legend()
      plt.xlabel('Epoch of training', fontsize=20)
      plt.ylabel('Error', fontsize=20)
      plt.show()
      
      return None
    
    def predict(self, X_batch):
      """
      Compute the output for instances in X_batch
      Returns: output probabilities
      """

      A = X_batch.T
      for l in range(0, self.n_layers + 1):
          Z = self.weights[l].dot(A) + self.biases[l]
          A = self.activation(Z)[0]

      return A
    
    def __feed_forward(self, X_batch, y_batch=None):
      """
      Performs a forward pass using batch X_batch
        - Feed the batch through each hidden layer, and compute the output vector
        - Compute the error between output and ground truth y_batch using cross entropy loss
      Parameters:
        X_batch: batch used in forward pass
        y_batch: labels for X_batch
      Returns:
        model error on batch, output probabilities
      """
      A = X_batch
      
      # Feed input signal through the hidden layers
      for l in range(0, self.n_layers + 1):
          Z = self.weights[l].dot(A) + self.biases[l]
          A = self.activation(Z)[0]

          self.Z[l] = Z
          self.df[l] = self.activation(Z)[1]
          self.A[l] = A
      
      # Compute the output
      # TODO
  
      # Compute the error
      if y_batch is None: return None, A

      error = self.error_cost(A, y_batch)[0]
      
      return error, A
    
    def __backward_pass(self, X_batch, y_batch, epoch, from_fit=False):
      """
      Perform gradient backpropagation
        (ASSUMES output softmax activation & cross-entropy cost)
        About biases update for batch size > 1:
        https://stats.stackexchange.com/questions/373163/how-are-biases-updated-when-batch-size-1
      Parameters:
        X_batch : batch used in forward pass
        y_batch : labels for X_batch
        epoch   : epoch # 
      Returns : None
      """
      delta = [None] * (self.n_layers + 1)
      dW = [None] * (self.n_layers + 1)
      db = [None] * (self.n_layers + 1)

      if not from_fit:
          self.__feed_forward(X_batch, y_batch)
      
      # Error on output layer
      y_hat = self.A[self.n_layers]
      delta[self.n_layers] = self.error_cost(y_hat, y_batch)[1] * self.df[self.n_layers]

      # Backpropagate the error in the hidden layers
      for l in range(self.n_layers-1, -1, -1):
          delta[l] = self.weights[l + 1].T.dot(delta[l + 1]) * self.df[l]


      # Calculate number of exemples in one batch
      nb = X_batch.shape[1]

      # Calculate changes
      for l in range(self.n_layers, 0, -1):
          dW[l] = delta[l].dot(self.A[l - 1].T) / nb
          db[l] = self.__ave_delta(delta[l])

      dW[0] = delta[0].dot(X_batch.T) / nb
      db[0] = self.__ave_delta(delta[0])


      # Update the parameters
      for l in range(0, self.n_layers + 1):
          self.weights[l] -= self.learning_rate * dW[l]
          self.biases[l] -= self.learning_rate * db[l]
      
      return None
    
    @staticmethod
    def __ave_delta(delta):
      return np.array([delta.mean(axis=1)]).transpose()
    
    def __str__(self):
      output = ""
      for l in range(0, self.n_layers+1): # +1: output layer
          output += "---- LAYER " + str(l) + "\n"
          output += "  * Weights"+ str(self.weights[l].shape) + "\n"
          output += str(self.weights[l]) + "\n"
          output += "  * Biases"+ str(self.biases[l].shape) + "\n"
          output += str(self.biases[l]) + "\n"
          if self.A[l] is not None:
              output += "  * Activations"+ str(self.A[l].shape) + "\n"
              output += str(self.A[l]) + "\n"
      return output
    
    def __repr__(self):
      return self.__str__()


###############################################################################
# MODULE TEST
###############################################################################
import pandas as pd

def main(): 
    # Load data
    root = ""
    iris_df = pd.read_csv(root + 'iris.csv')
    
    # Extrate attributes names and target
    df_columns = iris_df.columns.values.tolist()
    features = df_columns[0:4]
    label    = df_columns[4:] # ['class']
    
    X = iris_df[features]
    y = iris_df[label]
    y = pd.get_dummies(y, dtype='int') # one-hot encoding
    
    X_train, X_val, y_train, y_val = \
      train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert dataframes to numpy arrays
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_val, y_val = X_val.to_numpy(), y_val.to_numpy()
    
    # Instanciate a neural network
    nn = NeuralNet(hidden_layer_sizes=(3,2), batch_size=4, activation='tanh',
                   learning_rate=0.01, epoch=100)
    
    # Fit the model
    nn.fit(X_train, y_train, X_val, y_val)
    
    return

if __name__ == '__main__':
    main()