import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numba import numba, jit

# Method to train the weights using the given data
def train_model(iterations, alpha, training_data, n_features, g_k, weights, n_xtra_rows, check_con):
        if not check_con:
            print("-------------------------------------------")
            print("Alpha: ", alpha, " Iterations: ", iterations)
        errors = []
        progress = 0
        for i in range(iterations):
            grad_W_MSE      = np.array(np.zeros((3, n_features)), dtype=np.float64)
            MSE_list    = np.array([[0]], dtype=np.float64)
            row_holder = np.array([np.zeros(n_features)])

            # Method seperated into a function that could be hardware accelerated
            grad_W_MSE, MSE_list = train_once(weights, training_data, grad_W_MSE, MSE_list, row_holder, g_k, n_xtra_rows)

            weights -= alpha*grad_W_MSE

            #
            current_progress = np.floor(10*i/iterations)
            if current_progress > progress:
                print(current_progress*10, "%")
                progress = current_progress
            errors.append(MSE_list[0, 0])
            if check_con: # We only check when exploring alphas, as this check adds significant runtime
                if not check_convergence(errors, MSE_list[0, 0]):
                    print("No convergance, run ended at: ", i, " iterations")
                    return "No convergence", errors
        return weights, errors


@jit(target_backend='cuda', nopython=True)
def train_once(weights, training_data, grad_W_MSE, MSE_list, row_holder, g_k, n_xtra_rows):
    for j, row in enumerate(training_data):
        row = np.append(row, [1]*n_xtra_rows)
        row_holder[0] = row
        ans = int(np.floor(j/30)) # We know this is the expected flower because of how the training data is created
        tk = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
        tk[ans] = 1
        new_row = np.array([[0.0]]*len(row), dtype=np.float64) # Workaround as np.reshape() could not be hardware accelerated
        for k in range(len(row)):
            new_row[k, 0] = row[k]
        g_k             = 1.0 / (1.0 + np.exp(-multiply_matrix(weights, new_row, "g_k")))
        MSE_list        = MSE_list + 0.5*multiply_matrix((g_k-tk).T, (g_k-tk), "MinSqError")
        grad_gk_mse     = np.multiply((g_k-tk), g_k)
        grad_W_MSE      = grad_W_MSE + multiply_matrix(np.multiply(grad_gk_mse, (1.0-g_k)), row_holder, "GradW")
    return grad_W_MSE, MSE_list


# A rather simle method of determining if the error is actually converging for the current alpha
@jit(target_backend='cuda', nopython=True)
def check_convergence(errors, error):
    typed_errors = numba.typed.List()
    [typed_errors.append(x) for x in errors]
    if (len(typed_errors) > 200):
        return error < (np.average(typed_errors)*0.99)
    return True


def run_training_and_tests(train_size, iterations, alpha, setosa, versicolor, virginica, n_types, n_features, n_xtra_rows, check_con):
    setosa_t        = setosa[0:train_size]
    versicolor_t    = versicolor[0:train_size]
    virginica_t     = virginica[0:train_size]
    training_data   = np.concatenate((setosa_t, versicolor_t, virginica_t), axis=0)
    testing_data_1    = np.array([setosa[train_size:], versicolor[train_size:], virginica[train_size:]])
    testing_data_2    = np.array([setosa[:train_size], versicolor[:train_size], virginica[:train_size]])
    g_k             = np.array([1., 0., 0.])
    weights         = np.array(np.zeros((n_types, n_features)))
    confusion_matrix_test = np.zeros((n_types, n_types))
    confusion_matrix_training = np.zeros((n_types, n_types))

    weights, errors = train_model(iterations, alpha, training_data, n_features, g_k, weights, n_xtra_rows, check_con)

    if type(weights) == type("hellu"):
        return "No convergence", errors, 0, 0
    confusion_matrix_test = test_model(testing_data_1, weights, confusion_matrix_test, n_xtra_rows)
    confusion_matrix_training = test_model(testing_data_2, weights, confusion_matrix_training, n_xtra_rows)

    return weights, errors, confusion_matrix_test, confusion_matrix_training


def test_model(test, weights, confusion_matrix, n_xtra_rows):
    for ans, test_set in enumerate(test):
        for row in test_set:
            row = np.append(row, [1]*n_xtra_rows)
            prediction = np.argmax(np.matmul(weights, row))
            confusion_matrix[ans, prediction] += 1
    return confusion_matrix


@jit(target_backend='cuda', nopython=True)
def multiply_matrix(A, B, whodis):
    r = len(A)
    c = len(B[0])
    C = np.zeros((r, c))
    for row in range(r): 
        for col in range(c):
            for elt in range(len(B)):
                C[row, col] = C[row, col] + A[row, elt] * B[elt, col]
    return C


def plot_errors(name, errors, alpha):
    plt.title(name + " - MSE development, alpha = " + str(round(alpha, 3)))
    plt.xlabel("Iterations")
    plt.ylabel("Mean square error")
    plt.plot(errors, "r")
    plt.savefig("Result-MSE_" + name + ".png", dpi=150)
    plt.show()


def plot_confusionmatrix(name, conf_matrix):
    fig, ax = plt.subplots()
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions')
    plt.ylabel('Actuals')
    plt.title(name + ' - Confusion Matrix')
    plt.savefig("Result-Cmatrix_" + name + ".png", dpi=150)
    plt.show()


def plot_histogram(tot_data):
    sns.set()
    sns.set_style("white")
    tot_data.columns = ['sepal_length','sepal_width','petal_length','petal_width','Flower']
    tot_data['Flower'] = pd.Categorical(tot_data['Flower'])

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    for col, ax in zip(tot_data.columns[:4], axs.flat):
        sns.histplot(data=tot_data, x=col, kde=True, hue='Flower', common_norm=False, legend=ax==axs[0,0], ax=ax, bins=20)
    plt.subplots_adjust(hspace=0.3, wspace=0.15, left=0.05, right=0.98, bottom=0.1, top=0.95)
    plt.legend()
    plt.savefig('Result-hist.png', dpi=150)
    plt.show()


def print_data(name, weights, confusion_matrix):
    print("\nCurrent weights: \n", weights)
    print("\nGives confusion matrix: \n", confusion_matrix)
    n_correct = 0
    for i in range(3):
        n_correct += confusion_matrix[i, i]
    print("\nAnd error: \n", round(100*(1 - n_correct / np.sum(confusion_matrix)), 3), "%")


def run_with_convergance_test(name, train_size, iterations, alpha, setosa, versicolor, virginica, n_types, n_features, n_xtra_rows, plot=True, check_con=False):
    # converging_alphas   = []
    # diverging_alphas    = []
    good_data = True
    weights, errors, confusion_matrix_test, confusion_matrix_training = run_training_and_tests(train_size, iterations, alpha, setosa, versicolor, virginica, n_types, n_features, n_xtra_rows, check_con)
    if type(weights) == type('string'):
        print("alpha ", round(alpha, 4), "did not converge... moving on.")
        good_data = False
        # diverging_alphas.append(alpha)
    else:
        print("alpha ", round(alpha, 4), "Converged!")
        # converging_alphas.append(alpha)
    if good_data and plot:
        print_data(name, weights, confusion_matrix_training)
        print_data(name, weights, confusion_matrix_test)
        # plot_errors(name, errors, alpha)
        # plot_confusionmatrix(name, confusion_matrix_training)