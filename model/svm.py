import numpy as np
import cvxopt
import cvxpy as cp
from sklearn.metrics.pairwise import rbf_kernel
import time
from datetime import timedelta

def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

def polynomial_kernel(x1, x2 ,d):
    c = 1
    return np.power(linear_kernel(x1, x2) + c, d)

def _rbf_kernel(x,y,sigma):
    gamma = 1/2*(sigma**2)
    res = rbf_kernel(x,y, gamma)
    return res

class SVM:
    def __init__(self, kernel="linear_kernel", C=None, degree=3, sigma=5):
        self.a = None
        self.sv = None
        self.svt = None
        self.b = None
        self.C = C

        self.kernel = kernel
        self.degree = degree
        self.sigma = sigma
        if(self.C is not None):
            self.C = float(self.C)

    def _kernel(self, X1, X2=None):
        if self.kernel == "linear_kernel":
            return linear_kernel(X1, X2)
        elif self.kernel == "polynomial_kernel":
            return polynomial_kernel(X1, X2, self.degree)
        elif self.kernel == "rbf_kernel":
            return _rbf_kernel(X1, X2, self.sigma)
        else:
            print("Invalid Kernel")
            raise InvalidKernelError()

    def _qpSolver(self, X, y):
        n, features = X.shape
        K = self._kernel(X, X)
        print("kernel computation")

        P1 = np.matrix(np.outer(y,y) * K)
        q1 = np.matrix(-np.ones((n, 1)))
        A1 = np.matrix(y)
        b1 = np.matrix(0.0)
        G1 = np.diag(np.ones(n) * -1)
        h1 = np.zeros(n)
        G2 = np.identity(n)
        h2 = np.ones(n) * self.C
        # Define and solve the CVXPY problem.
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P1) + q1.T@x),
                         [G1@x <= h1,
                          G2@x <= h2,
                          A1@x == b1])
        t1 = time.time()
        print("solver start")
        prob.solve()
        t2 = time.time()
        print('Elapsed time: {}'.format(timedelta(seconds=t2-t1)))
        print(x.value)

        # P = cvxopt.matrix(np.outer(y,y) * K)
        # q = cvxopt.matrix(np.ones(n) * -1)
        # A = cvxopt.matrix(y, (1,n), 'd')
        # b = cvxopt.matrix(0.0)
        # if self.C is None:
        #     G = cvxopt.matrix(np.diag(np.ones(n) * -1))
        #     h = cvxopt.matrix(np.zeros(n))
        # else:
        #     G = cvxopt.matrix(np.vstack((np.diag(np.ones(n) * -1), np.identity(n))))
        #     h = cvxopt.matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))
        # cvxopt.solvers.options['show_progress'] = False
        # cvxopt.solvers.options['abstol'] = 1e-10
        # cvxopt.solvers.options['reltol'] = 1e-10
        # cvxopt.solvers.options['feastol'] = 1e-10
        # t1 = time.time()
        # res = cvxopt.solvers.qp(P, q, G, h, A, b)
        # t2 = time.time()
        # print('Elapsed time: {}'.format(timedelta(seconds=t2-t1)))
        # a = np.ravel(res['x'])
        return x.value, K

    def fit(self, X, y, truth_label=True):
        y = np.array(y)
        X = np.array(X)
        if set(y) != set([-1, 1]):
            _re_label = lambda x: np.array([1 if i==truth_label else -1 for i in x])
            y = _re_label(y)
        a, K = self._qpSolver(X, y)
        cond = a > 1e-4
        self.a = a[cond]
        self.sv = X[cond]
        self.svt = y[cond]
        self.sv_idx = np.where(cond)[0]
        self.b = 0
        self.sv = np.array(self.sv)
        n = len(self.svt)
        print(n)
        for idx in range(n):
            aw = sum([self.a[j]*self.svt[j]*K[self.sv_idx[idx]][self.sv_idx[j]] for j in range(n)])
            self.b += self.svt[idx]-aw
        self.b /= len(self.a)

    def predict(self, X):
        y_predict = np.zeros(len(X))
        K = self._kernel(X, self.sv)
        for idx in range(len(X)):
            # a = 0
            # for j in range(len(self.sv)):
            #     a += self._kernel(X[idx], self.sv[j])*self.a[j]*self.svt[j]
            # y_predict[idx]=a
            y_predict[idx] = sum(K[idx]*self.a*self.svt)
        return np.sign(y_predict + self.b), y_predict + self.b
