import numpy as np

class KNN():
    
    def __init__(self, k=3):
        self.k = k
        self.X_train = np.array([])
        self.y_train = np.array([])
    
    def euclidean_dist(self, v1, v2):
        return np.sqrt(sum((v1 - v2)**2))
    
    def evaluate(self, y_hat, y_real):
        return sum(y_hat == y_real) / y_hat.shape[0]
    
    
    def fit(self, X_train, y_train):
        # 
        self.X_train = X_train
        self.y_train = y_train
        
        return self

    def predict(self, X_real):
        y_hat = []
        for v1 in X_real:
            dist_list = []
            for index, v2 in enumerate(self.X_train):
                dist_list.append([index, self.euclidean_dist(v1,v2)])
                sl = sorted(dist_list,key=lambda x: x[1])
            sl = sl[:self.k]
            neighbors = []
            for i in range(len(sl)):
                neighbors.append(self.y_train[sl[i][0]])
            pred = max(set(neighbors), key=neighbors.count)
            y_hat.append(pred)
            return np.array(y_hat)

