import numpy as np
from numpy import linalg as LA
from numpy.linalg import matrix_rank
import math
import matplotlib.pyplot as plt
import math
import time
from scipy.sparse.linalg import svds

iterations = 100


class Ranks_Matrix:
    "builds the data, partition for train and test"
    def __init__(self, file_to_read, mode):
        self.mode = mode
        if self.mode == '100k':
            self.gold_ranks_matrix = np.zeros(shape=(943, 1682))
            self.train_rank_matrix = np.zeros(shape=(628, 1682))
            self.test_rank_matrix = np.zeros(shape=(315, 1682))
            self.deli = '\t'
        else:
            self.gold_ranks_matrix = np.zeros(shape=(6040, 3952))
            self.train_rank_matrix = np.zeros(shape=(4027, 3952))
            self.test_rank_matrix = np.zeros(shape=(2013, 3952))
            self.deli = '::'
        self.file = file_to_read

    def build_data(self):
        with open(self.file, 'r') as file:
            for line in file:
                line_split = line.split(self.deli)
                user_id = int(line_split[0])-1
                item_id = int(line_split[1])-1
                rank = int(line_split[2])
                self.gold_ranks_matrix[user_id][item_id] = rank
                "an entry 0 means the user did not rate the movie"
        np.random.shuffle(self.gold_ranks_matrix) #shuffle data
        if self.mode == '100k':
            self.train_rank_matrix, self.test_rank_matrix = self.gold_ranks_matrix[:628, :], self.gold_ranks_matrix[
                                                                                             628:, :]
        else:
            self.train_rank_matrix, self.test_rank_matrix = self.gold_ranks_matrix[:4027, :], self.gold_ranks_matrix[
                                                                                              4027:, :]


class Non_Convex_Project:
    "Non-convex approach 1 - projecting onto explicit rank constraint: minX∈Rm×n: rank(X)≤r f(x)"
    def __init__(self, train, test, iter_num, rank):
        self.train_matrix = train
        self.test_matrix = test
        self.rank = rank
        self.u_gold, self.sigma_gold, self.v_gold = svds(self.train_matrix, k=200)
        u, s, v = svds(self.u_gold.dot(self.v_gold), k=self.rank)
        self.X_train = np.dot(u, np.dot(np.diag(s), v))
        self.iter_num = iter_num
        self.beta = math.pow(self.sigma_gold[0], 2)
        self.error_train = []
        self.error_test = []

    def projected_gradient_descent(self):
        step_size = 1
        # step_size = 1 / self.beta
        for i in range(self.iter_num):
            gradient = np.zeros(np.shape(self.X_train))
            gradient[np.nonzero(self.train_matrix)] =self.X_train[np.nonzero(self.train_matrix)] - self.train_matrix[np.nonzero(self.train_matrix)]
            u, s, v = svds(self.X_train - step_size * gradient, k=self.rank)
            self.X_train = np.dot(u, np.dot(np.diag(s), v))
            self.error_train.append(0.5 * np.sum(np.square(
                self.X_train[np.nonzero(self.train_matrix)] - self.train_matrix[np.nonzero(self.train_matrix)])))
            self.error_test.append(0.5 * np.sum(np.square(
                    self.X_train[np.nonzero(self.test_matrix)] - self.test_matrix[np.nonzero(self.test_matrix)])))


class Non_Convex_factorized:
    "Non-convex approach 2 - gradient descent over factorized form: minU∈Rm×r,V∈Rn×r f(U)"
    def __init__(self, train, test, iter_num, t):
        self.train_matrix = train
        self.test_matrix = test
        self.t = t
        u_gold, sigma_gold, v_gold = svds(self.train_matrix, k=200)
        self.beta = math.pow(sigma_gold[0], 2)
        self.iter_num = iter_num
        self.error_train = []
        self.error_test = []

    def gradient_descent(self):
        #step_size = 0.001
        step_size = 1/self.beta
        u_gold, sigma_gold, v_gold = svds(self.train_matrix, k=200)
        u, _, v = svds(u_gold.dot(v_gold), k=self.t)
        v = v.T
        x_temp = np.dot(u, v.T)
        for i in range(self.iter_num):
            x_curr = x_temp
            grad_f = np.zeros(np.shape(x_curr))
            grad_f[np.nonzero(self.train_matrix)] = x_curr[np.nonzero(self.train_matrix)] - self.train_matrix[np.nonzero(self.train_matrix)]
            grad_u = np.dot(grad_f, v)
            grad_v = np.dot(grad_f.T, u)
            u = u - step_size * grad_u
            v = v - step_size * grad_v
            x_temp = np.dot(u, v.T)
            self.error_train.append(0.5 * np.sum(np.square(
                x_temp[np.nonzero(self.train_matrix)] - self.train_matrix[np.nonzero(self.train_matrix)])))
            self.error_test.append(0.5 * np.sum(np.square(
                x_temp[np.nonzero(self.test_matrix)] - self.test_matrix[np.nonzero(self.test_matrix)])))


class Convex_Relaxation:
    "Convex relaxation approach: use conditional gradient for solving minX∈Rm×n: kXk∗≤τ f(x)"
    def __init__(self, train, test, iter_num):
        self.train_matrix = train
        self.test_matrix = test
        self.u_gold, self.sigma_gold, self.v_gold = svds(self.train_matrix, k=100)
        self.X_train = self.u_gold.dot(self.v_gold)
        self.tau = np.sum(self.sigma_gold)
        self.iter_num = iter_num
        self.error_train = []
        self.error_test = []

    def conditional_gradient_method(self):
        for i in range(self.iter_num):
            print('con GD iter: ' + str(i))
            step_size = 2 / (2 + i)
            gradient = np.zeros(np.shape(self.X_train))
            gradient[np.nonzero(self.train_matrix)] = self.X_train[np.nonzero(self.train_matrix)] - self.train_matrix[
                np.nonzero(self.train_matrix)]
            v = self.find_v(gradient)  # solve optimization problem
            self.X_train += step_size * (v - self.X_train)
            self.error_train.append(0.5 * np.sum(np.square(
                self.X_train[np.nonzero(self.train_matrix)] - self.train_matrix[np.nonzero(self.train_matrix)])))
            self.error_test.append(0.5 * np.sum(np.square(
                    self.X_train[np.nonzero(self.test_matrix)] - self.test_matrix[np.nonzero(self.test_matrix)])))

    def find_v(self, X):
        u, s, v = LA.svd(X)
        sigma = math.ceil(s[0])
        m, n = X.shape[0], X.shape[1]
        A = np.zeros(shape=(m + n, m + n))
        for i in range(m + n):
            A[i][i] = sigma
        for row in range(m):
            for column in range(n):
                A[row][m + column] = -X[row][column]

        for row in range(n):
            for colum in range(m):
                A[m + row][colum] = -  np.transpose(X)[row][colum]
        w = np.ones(shape=(m + n, 1))
        for t in range(20):
            w = A.dot(w) / (LA.norm(A.dot(w)))
        u = w[:m] / (LA.norm(w[:m]))
        v = w[m:] / (LA.norm(w[m:]))
        return self.tau * u.dot(np.transpose(v))


class Graph:
    def __init__(self, iter_num,  non_conv_1, non_conv_2, conv):
        self.iter_num = iter_num
        self.non_conv_1 = non_conv_1
        self.non_conv_2 = non_conv_2
        self.conv = conv

    def plot_graph(self):
        "ploting"
        x_plot = np.arange(self.iter_num)
        plt.plot(x_plot, self.non_conv_1)
        plt.plot(x_plot, self.non_conv_2)
        plt.plot(x_plot, self.conv)
        plt.grid(True)
        plt.legend(['Projected GD', 'Factorized GD', 'Conditional GD'], loc='upper right')
        plt.savefig('plot1.png')
        plt.show()

        x_plot = np.arange(10, self.iter_num)
        plt.plot(x_plot, self.non_conv_1[10:])
        plt.plot(x_plot, self.non_conv_2[10:])
        plt.plot(x_plot, self.conv[10:])
        plt.grid(True)
        plt.legend(['Projected GD', 'Factorized GD', 'Conditional GD'], loc='upper right')
        plt.savefig('plot2.png')
        plt.show()


start = time.time()
M = Ranks_Matrix('u.data', '100k')
M.build_data()

non_conv_1 = Non_Convex_Project(M.train_rank_matrix, M.test_rank_matrix, iterations, 10)
non_conv_1.projected_gradient_descent()

non_conv_2 = Non_Convex_factorized(M.train_rank_matrix, M.test_rank_matrix, iterations, 10)
non_conv_2.gradient_descent()

conv_relax = Convex_Relaxation(M.train_rank_matrix, M.test_rank_matrix, iterations)
conv_relax.conditional_gradient_method()

graph = Graph(iterations, non_conv_1.error_test, non_conv_2.error_test, conv_relax.error_test)
graph.plot_graph()

end = time.time()
print('time for 100k: ' + str((end - start)/60) + 'minutes')


start2 = time.time()
M_2 = Ranks_Matrix('ratings.dat', '1M')
M_2.build_data()
non_conv_1m = Non_Convex_Project(M_2.train_rank_matrix, M_2.test_rank_matrix, iterations, 15)
non_conv_1m.projected_gradient_descent()

non_conv_2m = Non_Convex_factorized(M_2.train_rank_matrix, M_2.test_rank_matrix, iterations, 15)
non_conv_2m.gradient_descent()

conv_relax_m = Convex_Relaxation(M_2.train_rank_matrix, M_2.test_rank_matrix, iterations)
conv_relax_m.conditional_gradient_method()

graph = Graph(iterations, non_conv_1m.error_test, non_conv_2m.error_test, conv_relax_m.error_test)
graph.plot_graph()

end2 = time.time()
print('time for 1M: ' + str((end2 - start2)/60) + 'minutes')