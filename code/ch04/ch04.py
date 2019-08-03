# coding: utf-8


import pandas as pd
from io import StringIO
import sys
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# *Python Machine Learning 2nd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Code Repository: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Code License: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Python Machine Learning - Code Examples

# # Chapter 4 - Building Good Training Sets – Data Preprocessing

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).





# *The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*


# ### Overview

# - [Dealing with missing data](#Dealing-with-missing-data)
#   - [Identifying missing values in tabular data](#Identifying-missing-values-in-tabular-data)
#   - [Eliminating samples or features with missing values](#Eliminating-samples-or-features-with-missing-values)
#   - [Imputing missing values](#Imputing-missing-values)
#   - [Understanding the scikit-learn estimator API](#Understanding-the-scikit-learn-estimator-API)
# - [Handling categorical data](#Handling-categorical-data)
#   - [Nominal and ordinal features](#Nominal-and-ordinal-features)
#   - [Mapping ordinal features](#Mapping-ordinal-features)
#   - [Encoding class labels](#Encoding-class-labels)
#   - [Performing one-hot encoding on nominal features](#Performing-one-hot-encoding-on-nominal-features)
# - [Partitioning a dataset into a separate training and test set](#Partitioning-a-dataset-into-seperate-training-and-test-sets)
# - [Bringing features onto the same scale](#Bringing-features-onto-the-same-scale)
# - [Selecting meaningful features](#Selecting-meaningful-features)
#   - [L1 and L2 regularization as penalties against model complexity](#L1-and-L2-regularization-as-penalties-against-model-omplexity)
#   - [A geometric interpretation of L2 regularization](#A-geometric-interpretation-of-L2-regularization)
#   - [Sparse solutions with L1 regularization](#Sparse-solutions-with-L1-regularization)
#   - [Sequential feature selection algorithms](#Sequential-feature-selection-algorithms)
# - [Assessing feature importance with Random Forests](#Assessing-feature-importance-with-Random-Forests)
# - [Summary](#Summary)






# # Dealing with missing data

# ## Identifying missing values in tabular data



#サンプルデータを作成
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# If you are using Python 2.7, you need
# to convert the string to unicode:

"""if (sys.version_info < (3, 0)):
    csv_data = unicode(csv_data)"""
#サンプルデータを読み込む
df = pd.read_csv(StringIO(csv_data))
df



#各特長量の欠損値をカウント
df.isnull().sum()




# access the underlying NumPy array
# via the `values` attribute
df.values



# ## Eliminating samples or features with missing values



# remove rows（行） that contain missing values

df.dropna(axis=0)




# remove columns（列） that contain missing values

df.dropna(axis=1)




"""# remove columns that contain missing values

df.dropna(axis=1)"""




# only drop rows where all columns are NaN

df.dropna(how='all')  




# drop rows that have less than 3 real values 
#非NaN値が4つ未満の行を削除
df.dropna(thresh=4)




# only drop rows where NaN appear in specific columns (here: 'C')
#特定の列"C"にNaNが含まれている行だけを削除
df.dropna(subset=['C'])



# ## Imputing missing values



# again: our original array
df.values




# impute missing values via the column mean

#欠損値補完のインスタンスを生成（平均値補完）
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
#データを適合
imr = imr.fit(df.values)
#補完を実行
imputed_data = imr.transform(df.values)
imputed_data



# ## Understanding the scikit-learn estimator API










# # Handling categorical data

# ## Nominal and ordinal features



#サンプルデータを生成（Tシャツの色・サイズ・価格・クラスラベル）
df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])
#列名を設定
df.columns = ['color', 'size', 'price', 'classlabel']
df



# ## Mapping ordinal features


#Tシャツのサイズと整数を対応させるディクショナリを生成
size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}
#Tシャツのサイズを整数に変換
df['size'] = df['size'].map(size_mapping)
df



#もとに戻す
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'].map(inv_size_mapping)



# ## Encoding class labels




# create a mapping dict
# to convert class labels from strings to integers
#Tシャツのサイズと整数を対応させるディクショナリを生成
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
class_mapping




# to convert class labels from strings to integers
#クラスラベルを整数に変換
df['classlabel'] = df['classlabel'].map(class_mapping)
df




# reverse the class label mapping
#もとに戻す
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df





# Label encoding with sklearn's LabelEncoder
#ラベルエンコーダのインスタンスを生成
class_le = LabelEncoder()
#クラスラベルから整数に変換
y = class_le.fit_transform(df['classlabel'].values)
y




# reverse mapping
#もとに戻す
class_le.inverse_transform(y)


# Note: The deprecation warning shown above is due to an implementation detail in scikit-learn. It was already addressed in a pull request (https://github.com/scikit-learn/scikit-learn/pull/9816), and the patch will be released with the next version of scikit-learn (i.e., v. 0.20.0).


# ## Performing one-hot encoding on nominal features


#Tシャツの色，サイズ，価格を抽出
X = df[['color', 'size', 'price']].values

color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X




#one-hot-encoderの生成
ohe = OneHotEncoder(categorical_features=[0])
#one-hotエンコーディングを実行
ohe.fit_transform(X).toarray()




# return dense array so that we can skip
# the toarray step

ohe = OneHotEncoder(categorical_features=[0], sparse=False)
ohe.fit_transform(X)




# one-hot encoding via pandas

pd.get_dummies(df[['price', 'color', 'size']])




# multicollinearity guard in get_dummies

pd.get_dummies(df[['price', 'color', 'size']], drop_first=True)




# multicollinearity guard for the OneHotEncoder
#one-hot-encoderの生成
ohe = OneHotEncoder(categorical_features=[0])
#one-hotエンコーディングを実行，列削除
ohe.fit_transform(X).toarray()[:, 1:]



# # Partitioning a dataset into a seperate training and test set


#wineデータセットを読み込む
df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)

# if the Wine dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

# df_wine = pd.read_csv('wine.data', header=None)

#列名を設定
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
#クラスラベルを表示
print('Class labels', np.unique(df_wine['Class label']))
#wineデータセットの先頭５行を表示
df_wine.head()




#特徴量とクラスラベルを別々に抽出
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
#トレーニングデータとテストデータに分割
#全体の30％をテストデータにする
X_train, X_test, y_train, y_test =    train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)



# # Bringing features onto the same scale


#正規化
#min-maxスケーリングのインスタンスを生成
mms = MinMaxScaler()
#トレーニングデータをスケーリング
X_train_norm = mms.fit_transform(X_train)
#テストデータをスケーリング
X_test_norm = mms.transform(X_test)




#標準化
#標準化のインスタンスを生成（平均＝0，標準偏差＝1に変換）
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


# A visual example:



ex = np.array([0, 1, 2, 3, 4, 5])

print('standardized:', (ex - ex.mean()) / ex.std())

# Please note that pandas uses ddof=1 (sample standard deviation) 
# by default, whereas NumPy's std method and the StandardScaler
# uses ddof=0 (population standard deviation)

# normalize
print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))



# # Selecting meaningful features

# ...

# ## L1 and L2 regularization as penalties against model complexity

# ## A geometric interpretation of L2 regularization









# ## Sparse solutions with L1-regularization





# For regularized models in scikit-learn that support L1 regularization, we can simply set the `penalty` parameter to `'l1'` to obtain a sparse solution:


#L1正則化ロジスティック回帰のインスタンスを生成
LogisticRegression(penalty='l1')


# Applied to the standardized Wine data ...



# L1正則化ロジスティクス回帰のインスタンスを生成（逆正則化パラメータC=1.0）
lr = LogisticRegression(penalty='l1', C=1.0)
# Note that C=1.0 is the default. You can increase
# or decrease it to make the regulariztion effect
# stronger or weaker, respectively.
# トレーニングデータに適合
lr.fit(X_train_std, y_train)
# トレーニングデータに対する正解率の表示
print('Training accuracy:', lr.score(X_train_std, y_train))
# テストデータに対する正解率の表示
print('Test accuracy:', lr.score(X_test_std, y_test))



# 切片の表示
lr.intercept_




np.set_printoptions(8)




lr.coef_[lr.coef_!=0].shape



# 重み係数の表示
lr.coef_



# 描画の準備
fig = plt.figure()
ax = plt.subplot(111)

# 各係数の色のリスト
colors = ['blue', 'green', 'red', 'cyan', 
          'magenta', 'yellow', 'black', 
          'pink', 'lightgreen', 'lightblue', 
          'gray', 'indigo', 'orange']

# 空のリストを生成（重み係数，逆正則パラメータ）
weights, params = [], []
# 逆正則パラメータの値ごとに処理
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

# 重み係数をNumPy配列に変換
weights = np.array(weights)

# 各重み係数をプロット
for column, color in zip(range(weights.shape[1]), colors):
    # 横軸を逆正則化パラメータ，縦軸を重み係数とした折れ線グラフ
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)
# y＝0に黒い波線を引く
plt.axhline(0, color='black', linestyle='--', linewidth=3)
# 横軸の範囲の設定
plt.xlim([10**(-5), 10**5])
# 軸ラベルの設定
plt.ylabel('weight coefficient')
plt.xlabel('C')
# 横軸を対数スケールに設定
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', 
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
#plt.savefig('images/04_07.png', dpi=300, 
#            bbox_inches='tight', pad_inches=0.2)
plt.show()



# ## Sequential feature selection algorithms





class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        
        X_train, X_test, y_train, y_test =             train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, 
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, 
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score





knn = KNeighborsClassifier(n_neighbors=5)

# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
# plt.savefig('images/04_08.png', dpi=300)
plt.show()




k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])




knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))




knn.fit(X_train_std[:, k3], y_train)
print('Training accuracy:', knn.score(X_train_std[:, k3], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k3], y_test))



# # Assessing feature importance with Random Forests




feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
#plt.savefig('images/04_09.png', dpi=300)
plt.show()





sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of features that meet this threshold criterion:', 
      X_selected.shape[1])


# Now, let's print the 3 features that met the threshold criterion for feature selection that we set earlier (note that this code snippet does not appear in the actual book but was added to this notebook later for illustrative purposes):



for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))



# # Summary

# ...

# ---
# 
# Readers may ignore the next cell.




