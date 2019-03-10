import numpy as np

class Kernel:
    def __init__(self, name='linear', para=None):
        self.name = name
        self.para = para

    def __call__(self, data1, data2):
        if self.name == 'linear':
            return data1 @ data2.T
        if self.name == 'poly':
            if self.para == None:
                print('please assign a integer to p')
            return (data1 @ data2.T+1)**self.para
        # rbf not usable yet.
#         if self.name=='rbf':
#             if self.para==None:
#                 print('please assign a integer to sigma')
#             return np.exp(-0.5*((data1[:,np.newaxis,:]-data2[np.newaxis,:,:]).sum(-1)/self.para)**2)


class SVM:
    def __init__(self, data, label, C=1, eps=1e-6, kernel=Kernel(), maxiter=200):
        self.X = data
        self.y = label
        self.C = C
        self.eps = eps
        self.Kernel = kernel
        self.K = self.Kernel(data, data)
        self.a = np.zeros(data.shape[0])
        self.b = 0
        self.maxiter = maxiter

    def calE(self, i):
        return (self.a * self.y) @ self.K[i] + self.b - self.y[i]

    def KKT(self, i):
        '''
        a = 0     =>  outside
        0 < a < c =>  on the margin, active support
        a = c     =>  inside
        $y_i E_i = y_i (g_i - y_i) = y_i g_i - 1$
        '''
        ye = self.y[i]*self.calE(i)
        ai = self.a[i]
        return np.isclose(ai, 0.0, atol=self.eps) and ye >= 0\
            or self.eps < ai < self.C and np.isclose(ye, 0.0, atol=self.eps)\
            or np.isclose(ai, self.C, atol=self.eps) and ye <= 0

    def select_ij(self):
        active = [] # active set
        others = []
        for i in range(self.X.shape[0]):
            if 0 < self.a[i] < self.C:
                active.append(i)
            else:
                others.append(i)
        # 1. select i,j
        # active set first, if failed, try others
        for i in active+others:
            if not self.KKT(i):
                Ei = self.calE(i)
                okj, okEj, maxv, oka2 = 0, 0, 0, 0
                # active set first, if failed, try others
                for idx, j in enumerate(active+others):
                    # 7.107
                    eta = self.K[i, i]+self.K[j, j]-2*self.K[i, j]
                    # page 126
                    L, H = max(0, self.a[j]-self.a[i]), min(self.C, self.C+self.a[j]-self.a[i])
                    if self.y[i] == self.y[j]:
                        L, H = max(0, self.a[j]+self.a[i]-self.C), min(self.C, self.a[j]+self.a[i])
                    # odd case
                    if L == H or eta <= 0:
                        continue
                    Ej = self.calE(j)
                    # 7.106
                    a2 = self.a[j]+self.y[j]*(Ei-Ej)/eta
                    a2 = np.clip(a2, L, H)
                    # get the max update value in the active set
                    delta = a2-self.a[j]
                    if abs(delta) > maxv:
                        okj, okEj, maxv, oka2 = j, Ej, abs(delta), a2
                    # if good step found in active, break
                    if idx+1 == len(active) and not np.isclose(maxv, 0, self.eps):
                        break
                if np.isclose(maxv, 0, self.eps):
                    continue
                return i, okj, Ei, okEj, oka2
        return None

    def fit(self):
        it = 0
        while it < self.maxiter:
            res = self.select_ij()
            if res == None:
                break
            i, j, Ei, Ej, a2 = res
            a1 = self.a[i]+self.y[i]*self.y[j]*(self.a[j]-a2)
            # 3. cal b (7.115-116)
            b1 = self.b-Ei-self.y[i]*self.K[i, i] * (a1-self.a[i])\
                          -self.y[j]*self.K[j, i]*(a2-self.a[j])
            b2 = self.b-Ej-self.y[i]*self.K[i, j] * (a1-self.a[i])\
                          -self.y[j]*self.K[j, j]*(a2-self.a[j])
            # select b (under 7.116)
            self.b = b1 if 0 < a1 < self.C else (
                     b2 if 0 < a2 < self.C else (b1+b2)/2)
            self.a[i] = a1
            self.a[j] = a2
            it += 1

        if it == self.maxiter:
            print("Max iter number exceeded!")
        else:
            print("Done!")
        return it

    def predict(self, data):
        return np.sign((self.a * self.y) @ self.Kernel(self.X, data) + self.b)

if __name__=="__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    iris = load_iris()
    data = iris["data"][:100, :2]
    label = np.hstack((np.repeat(-1, 50), np.repeat(1, 50)))

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.25)

    model = SVM(X_train, y_train, maxiter=200)
    model.fit()

    res = model.predict(data)
    err = res-label
    acc=err[err == 0].size/res.size*100
    print("Acc: %.2f %%"%acc)
