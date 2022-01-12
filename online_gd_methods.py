import scipy.io as sio
import numpy as np
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import copy
import time


def load_matrix():
    """"builds the matrix of rt"""
    mat = sio.loadmat('data_490_1000.mat')
    matrix = mat['A']
    data = np.zeros(shape=(matrix.shape[0]*2, matrix.shape[1]))
    num_of_stocks = matrix.shape[0]
    for stock in range(matrix.shape[0]):
        for iter in range(matrix.shape[1]):
            if iter == 0:
                data[stock][iter] = matrix[stock][iter]
                data[stock + num_of_stocks][iter] = matrix[stock][iter] #what should it be????????
            else:
                data[stock][iter] = matrix[stock][iter] / matrix[stock][iter -1]
                data[stock + num_of_stocks][iter] = matrix[stock][iter -1] / matrix[stock][iter]
    return data


def online_GD(data, iterations):
    """"Online gradient descent"""
    x = np.ones(data.shape[0])
    x = (1/data.shape[0]) * x
    wealth_values = []
    prev_wealth = 1
    G = math.sqrt(980)
    step_size = math.sqrt(2) / (G * math.sqrt(iterations))
    #step_size = 0.1
    print('step size OGD: ' + str(step_size))
    for i in range(iterations):
        r_temp = data.T[i]
        if i == 0:
            wealth_values.append(prev_wealth)
        else:
            wealth_values.append(prev_wealth * np.dot(x, r_temp))
            prev_wealth = prev_wealth * np.dot(x, r_temp)
        gradient = (-r_temp)/(np.dot(x, r_temp))
        x = projection_GD(x - step_size*gradient)
    return wealth_values


def projection_GD(v, z=1):
    """Projection onto unit simplex"""
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def online_exp_grad(data, iterations):
    """"Online RFTL with exp. step"""
    x = np.ones(data.shape[0])
    x = (1/data.shape[0]) * x
    wealth_values = []
    G = find_G(data)
    step_size = math.sqrt(math.log(980)) / (G * math.sqrt(2 * iterations))
    #step_size = 40
    print('G for EGD: ' + str(G))
    print('step size EGD: ' + str(step_size))
    prev_wealth = 1
    for i in range(iterations):
        r_temp = data.T[i]
        if i == 0:
            wealth_values.append(prev_wealth)
        else:
            wealth_values.append(prev_wealth * np.dot(x, r_temp.T))
            prev_wealth = prev_wealth * np.dot(x, r_temp.T)
        gradient = (-r_temp) / (np.dot(x, r_temp.T))
        denominator = 0
        for index in range(x.shape[0]):
            denominator += x[index] * math.exp(-step_size*gradient[index])
        for index in range(x.shape[0]):
            x[index] *= math.exp(-step_size * gradient[index])
        x /= denominator
    return wealth_values


def online_newton_step(data, iterations):
    """Online newton step"""
    x = np.ones(data.shape[0])
    x = (1/data.shape[0]) * x
    wealth_values = []
    G = math.sqrt(980)
    gamma = 0.5 * min(1 / (4 * G * math.sqrt(2)), 1)
    epsilon = 1 / (2 * math.pow(gamma, 2))
    A = np.identity(data.shape[0]) * epsilon
    step_size = 1 / gamma
    #step_size = 210
    print('epsilon: ' + str(epsilon))
    print('step size ONS: ' + str(step_size))
    prev_wealth = 1
    for i in range(iterations):
        r_temp = data.T[i]
        if i == 0:
            wealth_values.append(prev_wealth)
        else:
            wealth_values.append(prev_wealth * np.dot(x, r_temp.T))
            prev_wealth = prev_wealth * np.dot(x, r_temp.T)
        gradient = (-r_temp) / (np.dot(x, r_temp.T))
        #rank-1 update
        A += np.dot(gradient.T, gradient)
        #Newton step and projection
        y_temp = x - (step_size * np.dot(np.linalg.inv(A), gradient.T)).T #need to be a vector
        #project y to get x
        x = projection_newton_2(y_temp, A)
    return wealth_values


"""def projection_newton(x, A):
    #Projection for newton step - did not use it (too long)
    z0 = np.ones(x.shape[0])
    z0 = (1 / x.shape[0]) * z0
    fun = lambda z: np.dot(np.dot((z - x), A), np.transpose(z - x))
    cons = ({'type': 'eq', 'fun': lambda z: np.sum(z) - 1})
    bnds = []
    for i in range(x.shape[0]):
        #bnds.append((0, None))
        bnds.append((0, 1))
    res = minimize(fun, z0, method='SLSQP', bounds=bnds, constraints=cons)
    return res.x"""

def projection_newton_2(x, A):
    """Projection onto unit simplex using PGD for newton step"""
    z = np.ones(x.shape[0])
    z = (1/x.shape[0]) * z
    step_size = 0.05
    for i in range(50):
        gradient = np.dot(A + A.T, z - x)
        z = projection_GD(z - step_size*gradient)
    return z


def print_graph(data, wealth_OGD, wealth_EGD, wealth_ONS, iterations):
    """Print graphs and find the best portfolio and best fixed stock"""
    wealth_best_portfolio = []
    wealth_best_fixed_stock = []
    print('wealth for Online Gradient Descent: ' + str(wealth_OGD[iterations-1]))
    print('wealth for Online Exponentiated Gradient: ' + str(wealth_EGD[iterations-1]))
    print('wealth for Online Newton Step: ' + str(wealth_ONS[iterations-1]))
    max_fix_index = find_best_fixed_stock(data)

    prev_wealth_fixed_portfolio = 1
    prev_wealth_fixed_stock = 1
    x_for_best_portfolio = find_best_portfolio(data)

    for i in range(iterations):
        r_temp = data.T[i]
        if i == 0:
            wealth_best_fixed_stock.append(prev_wealth_fixed_stock)
            wealth_best_portfolio.append(prev_wealth_fixed_portfolio)
        else:
            wealth_best_fixed_stock.append(prev_wealth_fixed_stock * r_temp[max_fix_index])
            prev_wealth_fixed_stock = prev_wealth_fixed_stock * r_temp[max_fix_index]

            wealth_best_portfolio.append(prev_wealth_fixed_portfolio * np.dot(x_for_best_portfolio, r_temp.T))
            prev_wealth_fixed_portfolio = prev_wealth_fixed_portfolio * np.dot(x_for_best_portfolio, r_temp.T)
    print('wealth for best fixed stock: ' +str(wealth_best_fixed_stock[iterations-1]))
    print('wealth for best fixed portfolio: ' + str(wealth_best_portfolio[iterations - 1]))

    x_plot = np.arange(1, iterations)
    plt.plot(x_plot, wealth_OGD[1:])
    plt.plot(x_plot, wealth_EGD[1:])
    plt.plot(x_plot, wealth_ONS[1:])
    plt.plot(x_plot, wealth_best_portfolio[1:])
    plt.plot(x_plot, wealth_best_fixed_stock[1:])
    plt.grid(True)
    plt.legend(['Online Gradient Descent', 'Online Exponentiated Gradient','Online Newton Step', 'Best Portfolio', 'Best Fixed Stock'], loc='upper left')
    #plt.legend(['Online Gradient Descent', 'Online Exponentiated Gradient', 'Online Newton Step'], loc='upper left')
    plt.savefig('plot1.png')
    plt.show()


def find_G(data):
    """Find the bound for G"""
    max_bound = 0
    for stock in range(data.shape[0]):
        r_temp = data[stock]
        temp = np.max(r_temp) / np.min(r_temp)
        if temp > max_bound:
            max_bound = copy.copy(temp)
    return max_bound


def find_best_portfolio(data):
    """best portfolio"""
    x = np.ones(data.shape[0])
    x = (1/data.shape[0]) * x
    step_size = 0.05
    #G = 999 * math.sqrt(980)
    #step_size = math.sqrt(2) / (G * math.sqrt(999))
    for i in range(1):
        gradient = 0
        for t in range(data.shape[1]):
            r_temp = data.T[t]
            gradient += (-r_temp) / (np.dot(x, r_temp.T))
            x = projection_GD(x - step_size * gradient)
    return x


def find_best_fixed_stock(data):
    """best fixed stock"""
    wealth_values_per_stock = np.ones(data.shape[0])
    for i in range(1, data.shape[1]):
        for stock in range(data.shape[0]):
            wealth_values_per_stock[stock] *= data[stock][i]
    max_index = np.argmax(wealth_values_per_stock)
    return max_index


start = time.time()
matrix = load_matrix()
wealth_GD = online_GD(matrix, 1000)
wealth_exp_grad = online_exp_grad(matrix, 1000)
wealth_newton_step = online_newton_step(matrix, 1000)
print_graph(matrix, wealth_GD, wealth_exp_grad, wealth_newton_step, 1000)
end = time.time()
print('time: ' + str((end - start)/60) + 'minutes')
