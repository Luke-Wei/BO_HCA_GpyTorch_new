import numpy as np
from pyDOE import *

from hyperparametertuning.mlp_torch import mlp_objective_function_Diabete


class Ackley:

    def __init__(self, dimension_x=50, dim=50, dy=0):

        self.dim = dim
        self.acdim = dimension_x
        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)
        self.lb[:dimension_x] = np.array([-32.768] * dimension_x)  
        self.ub[:dimension_x] = np.array([32.768] * dimension_x)  
        self.dy = dy
        self.x_star = list([np.repeat(0, dimension_x).reshape(1, -1)])
        self.f_star = 0
        self.mean = 0
        self.std = 1
        self.f_star = (self.f_star - self.mean) / self.std

    def __call__(self, xx):

        val = self.evaluate_true(xx)
        if self.dy == 'Heter':
            std = griewank(xx)
            noise = np.random.normal(0, std, 1)
        else:
            noise = np.random.normal(0, self.dy, 1)
        val = val + noise
        val = val.item()
        return val

    def evaluate_true(self, xx):  

        xx = xx[:self.acdim]
        a = 20
        b = 0.2
        c = 2 * np.pi
        sum1 = np.mean(xx ** 2)
        sum2 = np.mean(np.cos(c * xx))
        term1 = -a * np.exp(-b * np.sqrt(sum1))
        term2 = -np.exp(sum2)
        R = term1 + term2 + a + np.exp(1)
        val = R.item()
        val = (val - self.mean) / self.std

        return val


class Branin:
    
    def __init__(self, dimension_x=2, dim=50, dy=0):
        self.dim = dim
        self.acdim = dimension_x
        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)
        self.lb[:dimension_x] = np.array([-5] * dimension_x)  
        self.ub[:dimension_x] = np.array([15] * dimension_x) 
        self.dy = dy
        self.x_star = list([np.repeat(0, dimension_x).reshape(1, -1)])
        self.f_star = 0.397887

        self.mean = 0
        self.std = 1
        self.f_star = (self.f_star - self.mean) / self.std        
        
        
    def __call__(self, xx):

        val = self.evaluate_true(xx)
        if self.dy == 'Heter':
            std = griewank(xx)
            noise = np.random.normal(0, std, 1)
        else:
            noise = np.random.normal(0, self.dy, 1)
        val = val + noise
        val = val.item()
        return val    
    
    def evaluate_true(self, xx):
        
        xx = xx[:self.acdim]
        a = 1
        b = 5.1 / (4 * (np.pi**2))
        c = 5 / np.pi        
        r = 6
        s = 10
        t = 1 / ( 8 * np.pi ) 
        
        x1 = xx[0]
        x2 = xx[1]
        
        term1 = a * (x2 - b * (x1**2) + c * x1 - r)**2
        term2 = s * (1-t) * np.cos(x1)
               
        R = term1 + term2 + s        
        val = R.item()
        val = (val - self.mean) / self.std
            
        return val


class Eggholder:  

    def __init__(self, dimension_x=2, dim=50, dy=0):

        self.dim = dim
        self.acdim = dimension_x
        self.lb = -1.17 * np.ones(dim)
        self.ub = 1.17 * np.ones(dim)
        self.dy = dy
        self.x_star = list([np.array([1, 0.7895]).reshape(1, -1)])
        self.f_star = - 959.6407
        self.mean = 1.96
        self.std = 347.31
        self.f_star = (self.f_star - self.mean) / self.std

    def __call__(self, xx):

        val = self.evaluate_true(xx)
        if self.dy == 'Heter':
            std = griewank(xx)
            noise = np.random.normal(0, std, 1)
        else:
            noise = np.random.normal(0, self.dy, 1)
        val = val + noise
        val = val.item()
        return val

    def evaluate_true(self, xx):

        xx = xx[:self.acdim]
        xx = xx * 512
        x1 = xx[0]
        x2 = xx[1]
        term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47)))
        term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
        R = term1 + term2
        val = R.item()
        val = (val - self.mean) / self.std
        return val


def griewank(x):
    x = np.atleast_2d(x) 
    d = x.shape[1]
    y = (np.sum(x ** 2 / 4000, axis=1) - np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))), axis=1) + 1) / d
    return y.reshape(-1)  


class Griewank_f:

    def __init__(self, dimension_x=50, dim=50, dy=0):

        self.dim = dim
        self.acdim = dimension_x
        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)
        self.lb[:dimension_x] = np.array([-50] * dimension_x)
        self.ub[:dimension_x] = np.array([50] * dimension_x)
        self.dy = dy
        self.x_star = list([np.repeat(0, dimension_x).reshape(1, -1)])
        self.f_star = 0
        self.mean = 0
        self.std = 1
        if dimension_x == 2:
            self.mean = 1.42
            self.std = 0.57
        elif dimension_x == 6:
            self.mean = 2.25
            self.std = 0.47
        self.f_star = (self.f_star - self.mean) / self.std

    def __call__(self, xx):

        val = self.evaluate_true(xx)
        if self.dy == 'Heter':
            std = griewank(xx)
            noise = np.random.normal(0, std, 1)
        else:
            noise = np.random.normal(0, self.dy, 1)
        val = val + noise
        val = val.item()
        return val

    def evaluate_true(self, xx):

        xx = xx[:self.acdim]
        d = self.acdim
        R = np.sum(xx ** 2 / 4000) - np.prod(np.cos(xx / np.sqrt(np.arange(1, d + 1)))) + 1
        val = R.item()
        val = (val - self.mean) / self.std
        return val


class Hartmann6:  

    def __init__(self, dimension_x=6, dim=50, dy=0):

        self.dim = dim
        self.acdim = dimension_x
        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)
        self.dy = dy
        self.x_star = list([np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]).reshape(1, -1)])
        self.f_star = - 3.32237
        self.mean = -0.26
        self.std = 0.38
        self.f_star = (self.f_star - self.mean) / self.std

    def __call__(self, xx):

        val = self.evaluate_true(xx)
        if self.dy == 'Heter': 
            std = griewank(xx)  
            noise = np.random.normal(0, std, 1)
        else:
            noise = np.random.normal(0, self.dy, 1)
        val = val + noise
        val = val.item() 
        return val

    def evaluate_true(self, xx):

        xx = xx[:self.acdim]
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])
        P = 10 ** (-4) * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                   [2329, 4135, 8307, 3736, 1004, 9991],
                                   [2348, 1451, 3522, 2883, 3047, 6650],
                                   [4047, 8828, 8732, 5743, 1091, 381]])

        outer = 0
        for ii in range(4):
            inner = 0
            for jj in range(6):
                xj = xx[jj]
                Aij = A[ii, jj]
                Pij = P[ii, jj]
                inner = inner + Aij * (xj - Pij) ** 2
            new = alpha[ii] * np.exp(-inner)
            outer = outer + new

        R = - outer
        val = R.item()
        val = (val - self.mean) / self.std
        return val


class Langermann:

    def __init__(self, dim=2, m=5, c=None, A=None):
        self.dim = dim
        self.m = m

        if c is None:
            if m == 5:
                self.c = np.array([1, 2, 5, 2, 3])
            else:
                raise ValueError("Value of the m-dimensional vector c is required.")
        else:
            self.c = np.array(c)

        if A is None:
            if m == 5 and dim == 2:
                self.A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])
            else:
                raise ValueError("Value of the (mxd)-dimensional matrix A is required.")
        else:
            self.A = np.array(A)

        self.mean = 0
        self.std = 1

    def __call__(self, xx):
        xx = np.array(xx)
        val = self.evaluate_true(xx)
        return val

    def evaluate_true(self, xx):
        if len(xx) != self.dim:
            raise ValueError(f"Input dimension mismatch: expected {self.dim}, got {len(xx)}")

        outer = 0
        for i in range(self.m):
            inner = 0
            for j in range(self.dim):
                xj = xx[j]
                Aij = self.A[i, j]
                inner += (xj - Aij) ** 2
            new = self.c[i] * np.exp(-inner / np.pi) * np.cos(np.pi * inner)
            outer += new

        val = outer
        val = (val - self.mean) / self.std  
        return val


class Levy:

    def __init__(self, dimension_x=50, dim=50, dy=0):

        self.dim = dim
        self.acdim = dimension_x
        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)
        self.lb[:dimension_x] = np.array([-10] * dimension_x)
        self.ub[:dimension_x] = np.array([10] * dimension_x)
        self.dy = dy
        self.x_star = list([np.repeat(1, dimension_x).reshape(1, -1)])
        self.f_star = 0
        self.mean = 0
        self.std = 1
        if dimension_x == 2:
            self.mean = 16.71
            self.std = 16.31
        elif dimension_x == 4:
            self.mean = 42.55
            self.std = 27.9
        self.f_star = (self.f_star - self.mean) / self.std

    def __call__(self, xx):

        val = self.evaluate_true(xx)
        if self.dy == 'Heter':
            std = griewank(xx)
            noise = np.random.normal(0, std, 1)
        else:
            noise = np.random.normal(0, self.dy, 1)
        val = val + noise
        val = val.item()
        return val

    def evaluate_true(self, xx): 

        xx = xx[:self.acdim]

        w = 1 + (xx - 1) / 4
        w1 = w[0]
        w2 = w[0:-1]
        w3 = w[-1]
        R = np.sin(np.pi * w1) ** 2 + (w3 - 1) ** 2 * (1 + np.sin(2 * np.pi * w3) ** 2) + np.sum(
            (w2 - 1) ** 2 * (1 + 10 * (np.sin(np.pi * w2 + 1) ** 2)))

        val = R.item()
        val = (val - self.mean) / self.std
        return val


class Schwefel: 

    def __init__(self, dimension_x=2, dim=50, dy=0):

        self.dim = dim
        self.acdim = dimension_x
        self.lb = -1 * np.ones(dim)
        self.ub = np.ones(dim)
        self.dy = dy
        self.x_star = list([np.repeat(0.8419, dimension_x).reshape(1, -1)])
        self.f_star = 0
        self.mean = 838.57
        self.std = 274.3
        if dimension_x == 10:
            self.mean = 4190.05
            self.std = 615.36
        self.f_star = (self.f_star - self.mean) / self.std

    def __call__(self, xx):

        val = self.evaluate_true(xx)
        if self.dy == 'Heter':
            std = griewank(xx)
            noise = np.random.normal(0, std, 1)
        else:
            noise = np.random.normal(0, self.dy, 1)
        val = val + noise
        val = val.item()
        return val

    def evaluate_true(self, xx):

        xx = xx[:self.acdim]
        xx = xx * 500
        term1 = np.sum(xx * np.sin(np.sqrt(np.abs(xx))))
        R = 418.9829 * self.acdim - term1
        val = R.item()
        val = (val - self.mean) / self.std
        return val


class MLP_Diabete:

    def __init__(self, dimension_x=4, dim=4, dy='nozero'):
        self.dim = dim
        self.acdim = dimension_x
        self.lb = np.array([0] * dimension_x)
        self.ub = np.array([1] * dimension_x)
        self.x_star = None
        self.f_star = 0

    def __call__(self, xx):
        print("xx",xx)
        val = mlp_objective_function_Diabete(xx, 1)
        val = val.item()

        return val

    def evaluate_true(self, xx):
        val = mlp_objective_function_Diabete(xx, 1)
        val = val.item()

        return val


class StyblinskiTang:

    def __init__(self, dimension_x=50, dim=50, dy=0):

        self.dim = dim
        self.acdim = dimension_x
        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)
        self.lb[:dimension_x] = np.array([-5] * dimension_x) 
        self.ub[:dimension_x] = np.array([5] * dimension_x)  
        self.dy = dy
        self.x_star = list([np.repeat(-2.903534, dimension_x).reshape(1, -1)])  
        self.f_star = -39.16617 * dimension_x / 2

        self.mean = 0
        self.std = 1
        self.f_star = (self.f_star - self.mean) / self.std

    def __call__(self, xx):

        val = self.evaluate_true(xx)
        if self.dy == 'Heter':
            std = np.sqrt(np.abs(val))  
            noise = np.random.normal(0, std, 1)
        else:
            noise = np.random.normal(0, self.dy, 1)
        val = val + noise
        val = val.item()
        return val

    def evaluate_true(self, xx):

        xx = xx[:self.acdim]
        term = xx ** 4 - 16 * xx ** 2 + 5 * xx
        R = np.sum(term) / 2
        val = R.item()
        val = (val - self.mean) / self.std

        return val
    

class Rastrigin:

    def __init__(self, dimension_x=50, dim=50, dy=0):
        self.dim = dim
        self.acdim = dimension_x
        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)
        self.lb[:dimension_x] = np.array([-5.12] * dimension_x) 
        self.ub[:dimension_x] = np.array([5.12] * dimension_x)  
        self.dy = dy
        self.x_star = list([np.repeat(0, dimension_x).reshape(1, -1)])
        self.f_star = 0

        self.mean = 0
        self.std = 1
        self.f_star = (self.f_star - self.mean) / self.std

    def __call__(self, xx):
        val = self.evaluate_true(xx)
        if self.dy == 'Heter':
            std = griewank(xx)  # Assuming a heteroscedastic noise function is defined elsewhere
            noise = np.random.normal(0, std, 1)
        else:
            noise = np.random.normal(0, self.dy, 1)
        val = val + noise
        val = val.item()
        return val

    def evaluate_true(self, xx):
        xx = xx[:self.acdim]
        d = len(xx)
        sum_term = np.sum(xx**2 - 10 * np.cos(2 * np.pi * xx))
        R = 10 * d + sum_term
        val = R.item()
        val = (val - self.mean) / self.std
        return val    
    

class Powell:

    def __init__(self, dimension_x=50, dim=50, dy=0):
        self.dim = dim
        self.acdim = dimension_x
        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)
        self.lb[:dimension_x] = np.array([-4] * dimension_x)
        self.ub[:dimension_x] = np.array([5] * dimension_x)
        self.dy = dy
        self.x_star = list([np.repeat(0, dimension_x).reshape(1, -1)])
        self.f_star = 0

        self.mean = 0
        self.std = 1
        self.f_star = (self.f_star - self.mean) / self.std

    def __call__(self, xx):
        val = self.evaluate_true(xx)
        if self.dy == 'Heter':
            std = griewank(xx) 
            noise = np.random.normal(0, std, 1)
        else:
            noise = np.random.normal(0, self.dy, 1)
        val = val + noise
        val = val.item()
        return val

    def evaluate_true(self, xx):
        xx = xx[:self.acdim]
        d = len(xx)
        sum_val = 0

        for ii in range(1, d // 4 + 1):
            term1 = (xx[4 * ii - 4] + 10 * xx[4 * ii - 3]) ** 2
            term2 = 5 * (xx[4 * ii - 2] - xx[4 * ii - 1]) ** 2
            term3 = (xx[4 * ii - 3] - 2 * xx[4 * ii - 2]) ** 4
            term4 = 10 * (xx[4 * ii - 4] - xx[4 * ii - 1]) ** 4
            sum_val += term1 + term2 + term3 + term4

        val = sum_val
        val = (val - self.mean) / self.std

        return val


class PermDB:

    def __init__(self, dimension_x=50, dim=50, dy=0, b=0.5):
        self.dim = dim
        self.acdim = dimension_x
        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)
        self.lb[:dimension_x] = np.array([-1.0] * dimension_x)  
        self.ub[:dimension_x] = np.array([1.0] * dimension_x)   
        self.dy = dy
        self.b = b
        self.x_star = list([np.repeat(0, dimension_x).reshape(1, -1)])
        self.f_star = 0

        self.mean = 0
        self.std = 1
        self.f_star = (self.f_star - self.mean) / self.std

    def __call__(self, xx):
        val = self.evaluate_true(xx)
        if self.dy == 'Heter':
            std = griewank(xx)
            noise = np.random.normal(0, std, 1)
        else:
            noise = np.random.normal(0, self.dy, 1)
        val = val + noise
        val = val.item()
        return val

    def evaluate_true(self, xx):
        xx = xx[:self.acdim]
        d = len(xx)
        outer = 0

        for ii in range(1, d + 1):
            inner = 0
            for jj in range(1, d + 1):
                xj = xx[jj - 1]
                inner += (jj ** ii + self.b) * ((xj / jj) ** ii - 1)
            outer += inner ** 2

        val = outer
        val = (val - self.mean) / self.std

        return val


class Rosenbrock:

    def __init__(self, dimension_x=50, dim=50, dy=0):

        self.dim = dim
        self.acdim = dimension_x
        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)
        self.lb[:dimension_x] = np.array([-5] * dimension_x) 
        self.ub[:dimension_x] = np.array([10] * dimension_x)  
        self.dy = dy
        self.x_star = [np.ones(dimension_x).reshape(1, -1)]
        self.f_star = 0

        self.mean = 0
        self.std = 1
        self.f_star = (self.f_star - self.mean) / self.std

    def __call__(self, xx):

        val = self.evaluate_true(xx)
        if self.dy == 'Heter':
            std = np.abs(np.mean(xx)) 
            noise = np.random.normal(0, std, 1)
        else:
            noise = np.random.normal(0, self.dy, 1)
        val = val + noise
        val = val.item()
        return val

    def evaluate_true(self, xx):

        xx = xx[:self.acdim]
        d = len(xx)
        total = 0
        for i in range(d - 1):
            xi = xx[i]
            xnext = xx[i + 1]
            total += 100 * (xnext - xi ** 2) ** 2 + (xi - 1) ** 2

        val = total
        val = (val - self.mean) / self.std
        return val

