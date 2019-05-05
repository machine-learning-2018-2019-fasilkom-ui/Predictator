import numpy as np
import cvxopt

def linear_kernel(x,y):
    # Dot Product
    return sum([x[i]*y[i] for i in range(len(x))])

def polynomial_kernel(x,y,d):
    c = 1
    return (linear_kernel(x, y) + c) **d

def rbf_kernel(x,y,sigma):
    e = np.e
    # Formula
    return e**(-1*(np.linalg.norm([x[i]-y[i] for i in range(len(x))])**2)/(2*(sigma**2)))

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

    def _kernel(self, v1, v2):
        if self.kernel == "linear_kernel":
            return linear_kernel(v1, v2)
        elif self.kernel == "polynomial_kernel":
            return polynomial_kernel(v1, v2, self.degree)
        elif self.kernel == "rbf_kernel":
            return rbf_kernel(v1, v2, self.sigma)
        else:
            print("Invalid Kernel")
            raise InvalidKernelError()

    def _qpSolver(self, X, y):
        n, features = X.shape

        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i,j] = self._kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n) * -1)
        A = cvxopt.matrix(y, (1,n), 'd')
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n) * -1))
            h = cvxopt.matrix(np.zeros(n))
        else:
            G = cvxopt.matrix(np.vstack((np.diag(np.ones(n) * -1), np.identity(n))))
            h = cvxopt.matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))

        cvxopt.solvers.options['show_progress'] = False
        res = cvxopt.solvers.qp(P, q, G, h, A, b)
        a = np.ravel(res['x'])
        return a

    def fit(self, X, y, truth_label=1):
        y = np.array(y)
        X = np.array(X)
        if set(y) != set([-1, 1]):
            _re_label = lambda x: [1 if i==truth_label else -1 for i in x]
            y = _re_label(y)

        a = self._qpSolver(X, y)
        self.a = []
        self.sv = []
        self.svt = []
        self.b = 0
        for idx, val in enumerate(a > 1e-4):
            if val:
                self.sv.append(X[idx])
                self.svt.append(y[idx])
                self.a.append(a[idx])
        self.sv = np.array(self.sv)
        n = len(self.svt)
        for idx in range(n):
            aw = sum([self.a[j]*self.svt[j]*self._kernel(self.sv[idx], self.sv[j]) for j in range(n)])
            self.b += self.svt[idx]-aw
        self.b /= len(self.a)

    def predict(self, X):
        y_predict = []
        for i in X:
            a = 0
            for j in range(len(self.sv)):
                a += self._kernel(i, self.sv[j])*self.a[j]*self.svt[j]
            y_predict.append(a)
        return np.sign(y_predict + self.b), y_predict + self.b
