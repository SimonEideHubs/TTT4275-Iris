import numpy as np
import pandas as pd
from auxillery import *
from timeit import default_timer as timer

iris_names      = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
features        = ['sepal_length','sepal_width','petal_length','petal_width']
n_types         = len(iris_names)
n_xtra_rows     = 1
n_features      = len(features)+n_xtra_rows
path_data               = 'iris.data'
path_setosa_data        = 'class_1'
path_versicolour_data   = 'class_2'
path_virginica_data     = 'class_3'

tot_data    = pd.read_csv(path_data)
setosa      = pd.read_csv(path_setosa_data) 
versicolor  = pd.read_csv(path_versicolour_data)
virginica   = pd.read_csv(path_virginica_data)

train_size  = 30
iterations  = 2000
alpha       = 0.01
alpha_tests = np.linspace(0.1, 0.0001, 20)


setosa      = setosa.to_numpy()
versicolor  = versicolor.to_numpy() 
virginica   = virginica.to_numpy()

# histogram_plot(tot_data)

start = timer()


name = "task 1a - normal"
print(name)
run_with_convergance_test(name, train_size, iterations, alpha, setosa, versicolor, virginica, n_types, n_features, n_xtra_rows, plot=True)
print("\nRuntime: ", timer()-start)


""" name = "task 1b - convergence"
for current_a in alpha_tests:
    run_with_convergance_test(name, train_size, iterations, current_a, setosa, versicolor, virginica, n_types, n_features, n_xtra_rows, True, True)
print("\nRuntime: ", timer()-start) """
# plot_histogram(tot_data)


name = "task 1d - reverse"
print(name)
setosa_r        = setosa[::-1]
versicolor_r    = versicolor[::-1]
virginica_r     = virginica[::-1]


run_with_convergance_test(name, train_size, iterations, alpha, setosa_r, versicolor_r, virginica_r, n_types, n_features, n_xtra_rows, plot=True)
print("\nRuntime: ", timer()-start)

name = "task 2a - drop 1"
print(name)
drop_column = 1
setosa_drop1        = np.delete(setosa,     drop_column, 1)
versicolor_drop1    = np.delete(versicolor, drop_column, 1)
virginica_drop1     = np.delete(virginica,  drop_column, 1)

run_with_convergance_test(name, train_size, iterations, alpha, setosa_drop1, versicolor_drop1, virginica_drop1, n_types, n_features-1, n_xtra_rows, plot=True)
print("\nRuntime: ", timer()-start)

name = "task 2b - drop 2"
print(name)
drop_column = 0
setosa_drop2        = np.delete(setosa_drop1,     drop_column, 1)
versicolor_drop2    = np.delete(versicolor_drop1, drop_column, 1)
virginica_drop2     = np.delete(virginica_drop1,  drop_column, 1)

run_with_convergance_test(name, train_size, iterations, alpha, setosa_drop2, versicolor_drop2, virginica_drop2, n_types, n_features-2, n_xtra_rows, plot=True)
print("\nRuntime: ", timer()-start)

name = "task 2b - drop 3"
print(name)
drop_column = 1
setosa_drop3        = np.delete(setosa_drop2,     drop_column, 1)
versicolor_drop3    = np.delete(versicolor_drop2, drop_column, 1)
virginica_drop3     = np.delete(virginica_drop2,  drop_column, 1)

run_with_convergance_test(name, train_size, iterations, alpha, setosa_drop3, versicolor_drop3, virginica_drop3, n_types, n_features-3, n_xtra_rows, plot=True)
print("\nRuntime: ", timer()-start)