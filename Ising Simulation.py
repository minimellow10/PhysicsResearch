#This is the code from the Jupyter Notebook on Ising Simulation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import math as m
import random

#matplotlib inline

def initialize_grid(dim):
    '''
    Create a dim by dim array of numbers chosen randomly from -1 or 1.
    '''
    grid = np.random.choice([-1,1], size = (dim,dim))
    
    return grid

def plot_grid(grid, title=None):
    '''
    Plot grid with -1 being black and 1 being white with an option to add a title.
    '''

    # Create discrete colormap
    cmap = colors.ListedColormap(['black', 'white'])
    bounds = [-2,0,2]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    # Make plot
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap = cmap, norm=norm)
    ax.set_title(title)

def compute_magnetization(grid):
    return np.sum(grid)

def energy_change(grid, i, j):
    '''
    Assums a flip of the spin at position (i,j) on the grid and computes the change of
    energy due to it.
    Returns the change in energy.
    '''
    n = grid.shape[0]
    if (i==0):
        left = n-1
    else:
        left = i-1
    if (i==n-1):
        right = 0
    else:
        right = i+1
    if (j==0):
        down = n-1
    else:
        down = j-1
    if (j==n-1):
        up = 0
    else:
        up = j+1
    dE = 2 * grid[i,j] * (grid[left,j] + grid[right,j] + grid[i,up] + grid[i,down])
    return dE

def random_spin(grid):
    '''
    Choose randomly the position of one entry in the array grid.
    Returns the location of the spin.
    '''
    n = grid.shape[0]
    x_index = random.randint(0, n-1)
    y_index = random.randint(0, n-1)
    return (x_index, y_index)

def spin_flip(grid, T):
    '''
    Picks randomly one spin on the grid and calculates the change in energy due to flipping the spin.
    If change is negative, flip the spin.
    If change is positive, flip it only with some probability.
    Returns the new grid with the spin potentially flipped.
    '''
    i, j = random_spin(grid)
    delta_E = energy_change(grid, i, j)
    if delta_E < 0:
        grid[i,j] = -grid[i,j]
    elif random.random() < m.exp(-delta_E/T):
        grid[i,j] = -grid[i,j]
    return grid

def ising_simulation(n, T, steps=100, plot=False):
    '''
    Simulate 2d Ising model.
    Inputs:
    - n: size of the square lattice is n by n
    - T: temperature
    - steps: number of flips the algorithm tries to make
    - plot: decide if the grid is plotted at the beginning and the end of the simulation
    Returns:
    - the final grid
    - if plot is True, the initial and final grids in two plots
    '''
    grid = initialize_grid(n)
    
    if plot==True:
        plot_grid(grid, title='Initial grid')
        
    for i in range(steps):
        for i in range(n):
            grid = spin_flip(grid, T)
            
    if plot==True:
        plot_grid(grid, title='Final grid')
        
    return grid

ising_simulation(50, 3, 3000, plot=True)

def generate_data(size, num_temp, temp_min=0.1, temp_max=5, repeat=1, max_iter=None):
    '''
    Generate data from simulating the Ising model at different temperatures.
    The temperatures are spread equally between temp_min and temp_max.
    
    Input:
    - size: the grid is size x size
    - num_temp: number of different temperatures to consider
    - temp_min: minimum temperature to take
    - temp_max: maximum temperature
    - repeat: repeat the calculation for each temperature this number of times
    - max_iter: number of time steps in the simulation of the Ising model (default is size^2)
    
    Output:
    - raw_X: list of the arrays obtained from simulating the Ising (there are num_temp*repeat elements)
    - X: (num_temp, size^2) array where each line is a vectorized version of the grid and every line is a different run
    - y: (num_temp, 1) array that says if the simulation is made above the critical temps (y=1) or below (y=0)
    '''
    
    if max_iter==None:
        max_iter = size**2

    raw_X = []
    X = np.zeros((num_temp*repeat, size**2))
    y = np.zeros((num_temp*repeat, 1))
    temps = np.linspace(temp_min, temp_max, num = num_temp)

    for i in range(repeat):
        for j in range(num_temp):
            grid = ising_simulation(size, temps[j], max_iter)
            raw_X.append(grid)
            X[i*num_temp+j,:] = grid.reshape(1,grid.size)
            y[i*num_temp+j,:] = (temps[j] > 2.269)
            print(i*num_temp+j, end="\r")
    
    return raw_X, X, y

raw_X, X, y = generate_data(size=25, num_temp=40, repeat=20)

raw_X_test, X_test, y_test = generate_data(size=25, num_temp=20, repeat=10)

i = 70
plot_grid(raw_X[i])
y[i]

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2

# Initialize list of training and test errors
train_errors = []
test_errors = []


# Loop to run model many times and average over results

for i in range(15):

    # Print iteration to make sure everything works
    print(i, end="\r")
    
    # Initialize model
    model = Sequential()

    # Add hidden layer with 32 units and rlu activation. Add output layer with sigmoid activation.
    model.add(Dense(25, activation='relu', kernel_regularizer=l2(0.1), input_dim=25**2))
    model.add(Dense(15, activation='relu', kernel_regularizer=l2(0.1)))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fit model
    model.fit(X, y, epochs=40, verbose=0)

    # Add training and test errors to lists
    train_errors.append(model.evaluate(X,y,verbose=0)[1])
    test_errors.append(model.evaluate(X_test,y_test,verbose=0)[1])
    
# Average errors
avg_train_error = np.mean(train_errors)
std_train_error = np.std(train_errors)
avg_test_error = np.mean(test_errors)
std_test_error = np.std(test_errors)

print('Average of training errors: '+str(avg_train_error))
print('Standard deviation of training errors: '+str(std_train_error))
print('Average of test errors: '+str(avg_test_error))
print('Standard deviation of test errors: '+str(std_test_error))

(model.predict(X_test)>0.5)==y_test

plot_grid(raw_X_test[13])
y[13]

def deep_nn(X, y, X_test, y_test, layer1=10, layer2=10, lambd=0.05, num_epochs=50):

    # Initialize list of training and test errors
    train_errors = []
    test_errors = []


    # Loop to run model many times and average over results

    for i in range(15):

        # Print iteration to make sure everything works
        print(i, end="\r")
        
        # Initialize model
        model = Sequential()

        # Add hidden layer with 32 units and rlu activation. Add output layer with sigmoid activation.
        model.add(Dense(layer1, activation='relu', kernel_regularizer=l2(lambd), input_dim=25**2))
        model.add(Dense(layer2, activation='relu', kernel_regularizer=l2(lambd)))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Fit model
        model.fit(X, y, epochs=num_epochs, verbose=0)

        # Add training and test errors to lists
        train_errors.append(model.evaluate(X,y,verbose=0)[1])
        test_errors.append(model.evaluate(X_test,y_test,verbose=0)[1])
    
    
    print('Average of test errors for '+str(layer1)+' units in layer 1, '+str(layer2)+' units in layer 2, a regularization of '
      +str(lambd)+' and '+str(num_epochs)+' epochs: '+str(np.mean(test_errors)))

for lambd in [0, 0.05, 0.1]:
            
        deep_nn(X,y,X_test,y_test, lambd=lambd)

for layer1 in [10, 15, 20]:
    
        deep_nn(X,y,X_test,y_test, layer1)

for num_epochs in [40, 50, 60]:
    
    deep_nn(X,y,X_test,y_test, num_epochs=num_epochs)

raw_X, X, y = generate_data(size=25, num_temp=45, repeat=25)

deep_nn(X, y, X_test, y_test, layer1=25, layer2=15, lambd=0.1, num_epochs=40)

raw_X, X, y = generate_data(size=25, num_temp=40, repeat=20, max_iter=10*25*25)

deep_nn(X, y, X_test, y_test, layer1=25, layer2=15, lambd=0.1, num_epochs=40)

raw_X_test, X_test, y_test = generate_data(size=25, num_temp=20, repeat=10, max_iter=10*25*25)
deep_nn(X, y, X_test, y_test, layer1=25, layer2=15, lambd=0.1, num_epochs=40)

plot_grid(raw_X_test[10])
y[10]

plot_grid(initialize_grid(25))

