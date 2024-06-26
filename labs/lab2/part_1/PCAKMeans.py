from sklearn.datasets import load_wine
import numpy as np 
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors


def get_kernel_function(kernel:str):
    # TODO: implement different kernel functions 
    if kernel == "rbf":
        def rbf(x1:np.ndarray, x2:np.ndarray):
            return np.exp(-np.linalg.norm(x1 - x2) ** 2)
        return rbf
    if kernel == 'linear':
        def linear(x1:np.ndarray, x2:np.ndarray):
            return x1.dot(x2)
        return linear
    if kernel == 'poly':
        def poly(x1:np.ndarray, x2:np.ndarray):
            return (x1.dot(x2) + 1) ** 2
        return poly
    if kernel == 'sigmoid':
        def sigmoid(x1:np.ndarray, x2:np.ndarray):
            return np.tanh(x1.dot(x2) + 1)
        return sigmoid
    if kernel == 'cosine':
        def cosine(x1:np.ndarray, x2:np.ndarray):
            return x1.dot(x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return cosine
    return None

class PCA:
    def __init__(self, n_components:int=2, kernel:str="rbf") -> None:
        self.n_components = n_components
        self.kernel_f = get_kernel_function(kernel)
        # ...

    def fit(self, X:np.ndarray):
        m = X.shape[0]  # 样本数
        # 计算核矩阵
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K[i, j] = self.kernel_f(X[i], X[j])
        
        # 中心化核矩阵
        one_n = np.ones((m, m)) / m
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        
        # 计算特征值和特征向量
        eig_vals, eig_vecs = np.linalg.eigh(K)
        # 对特征值进行排序
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(m)]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        
        # 选取前k个特征向量
        self.alphas = np.column_stack([eig_pairs[i][1] for i in range(self.n_components)])
        
        return self
        

    def transform(self, X:np.ndarray):
        # X: [n_samples, n_features]
        # X_reduced = np.zeros((X.shape[0], self.n_components))
        # TODO: transform the data to low dimension
        X_reduced = np.zeros((X.shape[0], self.n_components))
        # 计算降维后的数据
        for i in range(X.shape[0]):
            for j in range(self.n_components):
                X_reduced[i, j] = np.sum(self.alphas[:, j] * np.array([self.kernel_f(X[i], X_k) for X_k in X]))
        return X_reduced

        

class KMeans:
    def __init__(self, n_clusters:int=3, max_iter:int=1000) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None
        self.labels = None

    # Randomly initialize the centers
    def initialize_centers(self, points):
        # points: (n_samples, n_dims,)
        n, d = points.shape

        self.centers = np.zeros((self.k, d))
        for k in range(self.k):
            # use more random points to initialize centers, make kmeans more stable
            random_index = np.random.choice(n, size=10, replace=False)
            self.centers[k] = points[random_index].mean(axis=0)
        
        return self.centers
    
    # Assign each point to the closest center
    def assign_points(self, points):
        # points: (n_samples, n_dims,)
        # return labels: (n_samples, )
        n_samples, n_dims = points.shape
        self.labels = np.zeros(n_samples)
        # TODO: Compute the distance between each point and each center
        # and Assign each point to the closest center
        for i in range(n_samples):
            # Compute the distance between each point and each center
            dists = np.linalg.norm(points[i] - self.centers, axis=1)
            # Assign each point to the closest center
            self.labels[i] = np.argmin(dists)

        return self.labels

    # Update the centers based on the new assignment of points
    def update_centers(self, points):
        # points: (n_samples, n_dims,)
        # TODO: Update the centers based on the new assignment of points
        for k in range(self.k):
            if points[self.labels == k].shape[0] == 0:
                continue
            self.centers[k] = points[self.labels == k].mean(axis=0)
        return self.centers

    # k-means clustering
    def fit(self, points):
        # points: (n_samples, n_dims,)
        # TODO: Implement k-means clustering
        self.k = self.n_clusters
        self.initialize_centers(points)
        for i in range(self.max_iter):
            labels = self.labels
            if labels is None:
                labels = np.zeros(points.shape[0])
            self.assign_points(points)
            centers = self.update_centers(points)
            if np.all(labels == self.labels):
                break
        return self

    # Predict the closest cluster each sample in X belongs to
    def predict(self, points):
        # points: (n_samples, n_dims,)
        # return labels: (n_samples, )
        return self.assign_points(points)
    
def load_data():
    words = [
        'computer', 'laptop', 'minicomputers', 'PC', 'software', 'Macbook',
        'king', 'queen', 'monarch','prince', 'ruler','princes', 'kingdom', 'royal',
        'man', 'woman', 'boy', 'teenager', 'girl', 'robber','guy','person','gentleman',
        'banana', 'pineapple','mango','papaya','coconut','potato','melon',
        'shanghai','HongKong','chinese','Xiamen','beijing','Guilin',
        'disease', 'infection', 'cancer', 'illness', 
        'twitter', 'facebook', 'chat', 'hashtag', 'link', 'internet',
    ]
    w2v = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary = True)
    vectors = []
    for w in words:
        vectors.append(w2v[w].reshape(1, 300))
    vectors = np.concatenate(vectors, axis=0)
    # print(len(words))
    return words, vectors

if __name__=='__main__':
    words, data = load_data()
    pca = PCA(n_components=2, kernel='cosine').fit(data)
    data_pca = pca.transform(data)

    kmeans = KMeans(n_clusters=7).fit(data_pca)
    clusters = kmeans.predict(data_pca)

    # plot the data
    
    plt.figure(figsize=(10, 10))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters)
    for i in range(len(words)):
        plt.annotate(words[i], data_pca[i, :]) 
    plt.title("PB21111639")
    plt.savefig("PCA_KMeans.png")