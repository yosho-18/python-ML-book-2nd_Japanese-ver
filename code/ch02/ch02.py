# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# *Python Machine Learning 2nd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Code Repository: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Code License: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Python Machine Learning - Code Examples

# # Chapter 2 - Training Machine Learning Algorithms for Classification

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).


# *The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*

# ### Overview
# 

# - [Artificial neurons – a brief glimpse into the early history of machine learning](#Artificial-neurons-a-brief-glimpse-into-the-early-history-of-machine-learning)
#     - [The formal definition of an artificial neuron](#The-formal-definition-of-an-artificial-neuron)
#     - [The perceptron learning rule](#The-perceptron-learning-rule)
# - [Implementing a perceptron learning algorithm in Python](#Implementing-a-perceptron-learning-algorithm-in-Python)
#     - [An object-oriented perceptron API](#An-object-oriented-perceptron-API)
#     - [Training a perceptron model on the Iris dataset](#Training-a-perceptron-model-on-the-Iris-dataset)
# - [Adaptive linear neurons and the convergence of learning](#Adaptive-linear-neurons-and-the-convergence-of-learning)
#     - [Minimizing cost functions with gradient descent](#Minimizing-cost-functions-with-gradient-descent)
#     - [Implementing an Adaptive Linear Neuron in Python](#Implementing-an-Adaptive-Linear-Neuron-in-Python)
#     - [Improving gradient descent through feature scaling](#Improving-gradient-descent-through-feature-scaling)
#     - [Large scale machine learning and stochastic gradient descent](#Large-scale-machine-learning-and-stochastic-gradient-descent)
# - [Summary](#Summary)


# # Artificial neurons - a brief glimpse into the early history of machine learning


# ## The formal definition of an artificial neuron


# ## The perceptron learning rule


# # Implementing a perceptron learning algorithm in Python

# ## An object-oriented perceptron API


# Perceptron
class Perceptron(object):
    """Perceptron classifier.　パーセプトロンの分類器

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.　トレーニングデータのトレーニング回数
    random_state : int
      Random number generator seed for random weight
      initialization.重みを初期化するための乱数シード

    Attributes　属性
    -----------
    w_ : 1d-array
      Weights after fitting.　適合後の重み
    errors_ : list
      Number of misclassifications (updates) in each epoch.　各エポックでの誤分類（更新）の数

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.　トレーニングデータに適応させる

        Parameters　パラメータ
        ----------
        X : {array-like　配列のようなデータ構造}, shape = [n_samples, n_features]
        　トレーニングデータ
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features（特徴量）.
        y : array-like, shape = [n_samples]
          Target values.　目的変数

        Returns　戻り値
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])  # 標準偏差0.01
        self.errors_ = []

        for _ in range(self.n_iter):  # トレーニング回数分トレーニングデータを反復
            errors = 0
            for xi, target in zip(X, y):  # 各サンプルで重み更新
                # 重み　w_1,...,w_mの更新
                # deltaw_j=eta(yi-yhati)xi_j(j=1,...,m)
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                # 重みw_0の更新，xi＝1
                self.w_[0] += update
                # 重みの更新が0でない場合は誤分類としてカウント
                errors += int(update != 0.0)
            # 反復回数ごとの誤差を格納
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input　総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step　１ステップ後のクラスラベルを返す"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


v1 = np.array([1, 2, 3])
v2 = 0.5 * v1
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# ## Training a perceptron model on the Iris dataset

# ...

# ### Reading-in the Iris data


df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data', header=None)
df.tail()

#
# ### Note:
# 
# 
# You can find a copy of the Iris dataset (and all other datasets used in this book) in the code bundle of this book, which you can use if you are working offline or the UCI server at https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data is temporarily unavailable. For instance, to load the Iris dataset from a local directory, you can replace the line 
# 
#     df = pd.read_csv('https://archive.ics.uci.edu/ml/'
#         'machine-learning-databases/iris/iris.data', header=None)
#  
# by
#  
#     df = pd.read_csv('your/local/path/to/iris.data', header=None)
# 


df = pd.read_csv('iris.data', header=None)
df.tail()

# ### Plotting the Iris data


# select setosa and versicolor
# 1－100行目の目的変数の抽出
y = df.iloc[0:100, 4].values
#Iris-setosaを‐1，Iris-virginicaを1に変換
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length，1－100行目の1，3列目の抽出，1列目はがく片の長さ，3列目は花びらの長さ
X = df.iloc[0:100, [0, 2]].values

# plot data
#品種setosaのプロット（赤の○）
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
#品種versicolorのプロット（青の×）
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

#軸ラベルの設定
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
#凡例の設定（左上に設置）
plt.legend(loc='upper left')

#図の表示
# plt.savefig('images/02_06.png', dpi=300)
plt.show()

# ### Training the perceptron model

#パーセプトロンのオブジェクトの生成（インスタンス化）
ppn = Perceptron(eta=0.1, n_iter=10)

#トレーニングデータへのモデルの適合
ppn.fit(X, y)

#エポックと誤分類誤差の関係の折れ線グラフをプロット
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')

plt.xlabel('Epochs')
plt.ylabel('Number of updates')

# plt.savefig('images/02_07.png', dpi=300)
plt.show()


# ### A function for plotting decision regions


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map，マーカーとカラーマップの準備
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface，決定領域のプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #グリッドポイントの生成
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    #各特長量を1次元配列に変換して予測を実行
    Q = np.array([xx1.ravel(), xx2.ravel()]).T#2*n行列
    Z = classifier.predict(Q)#１次元
    #予測結果を元のグリッドポイントのデータサイズに変換
    Z = Z.reshape(xx1.shape)
    #グリッドポイントの等高線のプロット
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    #軸の範囲の設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples，クラスごとにサンプルをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

# plt.savefig('images/02_08.png', dpi=300)
plt.show()


# # Adaptive linear neurons and the convergence of learning

# ...

# ## Minimizing cost functions with gradient descent


# ## Implementing an adaptive linear neuron in Python


class AdalineGD(object):
    """ADAptive LInear NEuron classifier（分類器）.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)，学習率
    n_iter : int
      Passes over the training dataset.
      トレーニングデータのトレーニング回数
    random_state : int
      Random number generator seed for random weight
      initialization.
      重みを初期化するための乱数シード


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
      １次元配列，適合後の重み
    cost_ : list
      Sum-of-squares cost function value in each epoch.
      リスト，各エポックでの誤差平方和のコスト関数

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            """activationメソッドは単なる恒等関数であるため，このコードは何の効果もないことに注意．
            代わりに，国設output=self.net_input(X)と記述することもできた．
            activationメソッドの目的は，より概念的なものである．つまり，（後ほど説明する）ロジスティクス回帰
            の場合は，ロジスティクス回帰の分類器を実装するためにシグモイド関数に変更することもできる"""
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(X)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,  
            # in the case of logistic regression (as we will see later), 
            # we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(net_input)
            #誤差y^(i) - Φ(z^(i))の計算
            errors = (y - output)
            #w_1,...,w_mの更新
            self.w_[1:] += self.eta * X.T.dot(errors)
            #w_0の更新　Δw_0=
            self.w_[0] += self.eta * errors.sum()
            #コスト関数の計算J(w)=1/2Σ_i(y^(i) - Φ(z^(i)))**2
            cost = (errors ** 2).sum() / 2.0
            #コストの格納
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step（１ステップ）"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

#描画領域を１行２列に分割
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
#勾配降下法によるADALINEの学習（学習率 eta=0.01）
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
#エポック数とコストの関係を表す折れ線グラフのプロット（縦軸のコストは常用対数）
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
#軸のラベルの設定
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
#タイトルの設定
ax[0].set_title('Adaline - Learning rate 0.01')

#勾配降下法によるADALINEの学習（学習率 eta=0.0001）
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

# plt.savefig('images/02_11.png', dpi=300)
plt.show()

# ## Improving gradient descent through feature scaling


# standardize features
#データのコピー
X_std = np.copy(X)
#各列の標準化
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

#勾配降下法によるADALINEの学習（学習率 eta=0.01）
ada = AdalineGD(n_iter=15, eta=0.01)
#モデルの適合
ada.fit(X_std, y)

#境界領域のプロット
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
#図の設定
plt.tight_layout()
# plt.savefig('images/02_14_1.png', dpi=300)
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')

plt.tight_layout()
# plt.savefig('images/02_14_2.png', dpi=300)
plt.show()


# ## Large scale machine learning and stochastic gradient descent


class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    shuffle : bool (default: True)
      Shuffles training data every epoch if True to prevent cycles.
      Trueの場合は，循環を回避するためにエポックごとにトレーニングデータをシャッフル
    random_state : int
      Random number generator seed for random weight
      initialization.
    　重みを初期化するための乱数シード

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value averaged over all
      training samples in each epoch.

        
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        #トレーニング回数分トレーニングデータを反復
        for i in range(self.n_iter):
            #指定された場合はトレーニングデータを反復
            if self.shuffle:
                X, y = self._shuffle(X, y)
            #各サンプルのコストを格納するリストの生成
            cost = []
            for xi, target in zip(X, y):
                #特長量xiと目的変数yを用いた重みの更新とコストの計算
                cost.append(self._update_weights(xi, target))
            #サンプルの平均コストの計算
            avg_cost = sum(cost) / len(y)
            #平均コストを格納
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):#オンライン学習
        """Fit training data without reinitializing the weights
        重みを再初期化することなくトレーニングデータに適合させる"""
        #初期化されていない場合は初期化を実行
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        #目的変数yの要素数が２以上の場合は
        #各サンプルの特長量x_iと目的変数targetで重みを更新
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        # 目的変数yの要素数が１の場合は
        #サンプル全体の特長量Xと目的変数yで重み更新
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers，重みを小さな乱数に初期化"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights
        ADALINEの学習規則を用いて重みを更新"""
        #活性化関数の出力の結果
        output = self.activation(self.net_input(xi))
        #誤差の計算
        error = (target - output)
        #重みw_1,...,w_mの更新
        self.w_[1:] += self.eta * xi.dot(error)
        #w_0の更新
        self.w_[0] += self.eta * error
        #コストの計算
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('images/02_15_1.png', dpi=300)
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')

plt.tight_layout()
# plt.savefig('images/02_15_2.png', dpi=300)
plt.show()

ada.partial_fit(X_std[0, :], y[0])

# # Summary

# ...

# --- 
# 
# Readers may ignore the following cell
