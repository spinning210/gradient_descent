#%%
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from numpy import arange



def target_func(x,get_points): 
    results = []
    min_x = 10
    min_y = 10
    for each in np.nditer(x):
        y = (1 - each*math.exp(-each))
        if y < min_y:
            min_y = y
            min_x = each
        results.append(y)
    
    np_results = np.array(results)
    if get_points == True:
        return np_results, min_x, min_y
    else :
        return np_results

def d_target_func(x): 
    reults = x*math.exp(-x)-math.exp(-x)
    return reults

def gradient_descent(x, iterations, learning_rate): 
    xs = np.zeros(iterations+1)
    xs[0] = x 
    for i in range(iterations):         
        gradient = d_target_func(x)      
        fix = - gradient * learning_rate        
        x += fix        
        xs[i+1] = x
    return xs

def init_par():
    x_start = 8  
    iterations = 30000
    learning_rate = 0.02
    original_plt = arange(0.0, 10.0, 0.01)

    return x_start, iterations, learning_rate, original_plt

def create_plt(original_plt,x,learning_rate):
    
    plt.plot(original_plt, target_func(original_plt, False), c = 'blue')
    results, min_x, min_y = target_func(x, True)
    plt.plot(x, results, c = 'red', label = 'learning rate={}'.format(learning_rate))   
    plt.legend()
    plt.show()
    print('min(x,y) = (',min_x, min_y,')')


x_start, iterations, learning_rate, original_plt = init_par()
x = gradient_descent(x_start, iterations, learning_rate)
create_plt(original_plt,x,learning_rate)




