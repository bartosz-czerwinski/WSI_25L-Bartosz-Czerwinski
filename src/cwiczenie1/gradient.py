import numpy as np
import autograd.numpy as anp
from autograd import grad
import plots

# Definicje funkcji
def f(x):
    return 10 * x ** 4 + 3 * x ** 3 - 30 * x ** 2 + 10 * x


def g(x1, x2):
    return (x1 - 2) ** 4 + (x2 + 3) ** 4 + 2 * ((x1 - 2) ** 2) * (x2 + 3) ** 2

def find_gradient(f):
    """
    Oblicza gradient funkcji f przy pomocy automatycznego różniczkowania.
    """
    return grad(lambda x: f(*x))


def gradient_descent(function, x0, learning_rate=0.003, tol=1e-3, max_iters=1000):
    x = anp.array(x0, dtype = float)
    trajectory = [x.copy()]
    grad_function = find_gradient(function)
    for i in range(max_iters):
        grad_val = grad_function(x)
        #Przerwanie obliczeń, jeśli gradient jest zbyt duży
        if np.linalg.norm(grad_val) > 1e100:
            return False
        #Przerwanie obliczeń, jeśli gradient jest mały (bardzo blisko minimum)
        if np.linalg.norm(grad_val) < tol:
            break
        x -= learning_rate * grad_val
        trajectory.append(x.copy())


    return x, np.array(trajectory)



results = gradient_descent(g, x0=[-4, -5.5], learning_rate=0.01, tol=1e-3, max_iters=10000)
#results = gradient_descent(f, x0=[2.5])
print(results[0])
print(results[0][1])
print(len(results[1]))

#results = gradient_descent(f, x0=[2], learning_rate=0.007, tol=1e-3, max_iters=10000)





plots.create_gif(g, results, x_range=(-8, 12), y_range=(-13, 7), resolution=1000)
#plots.create_gif(f, results, x_range=(-2.5, 2.5), resolution=1000)
#plots.plot_function(f, x_range=(-3.5, 3.5), resolution=1000, result_range=(-50, 20))
#plots.plot_function(g, x_range=(-8, 12), y_range=(-13, 7), resolution=1000, result_range=(-5, 1000))
