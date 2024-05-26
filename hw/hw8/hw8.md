## 1.
假设特征向量的维度为k，第i个特征值的取值种类数为$N_i$，那么我们可以创建一个有k层的决策树树，其中第i层（根节点为第1层）的所有节点为第i个特征值的决策节点，其中每个决策节点都有$N_i$个子节点，每个子节点代表第i个特征值的一个取值。那么这棵树的叶子节点的数量为$\prod_{i=1}^k N_i$，即每一种可能的特征向量都有一个叶子节点与之对应。因此，我们可以将训练集中的每个特征向量对应的叶节点的类别设置为该特征向量的类别，由于不含冲突数据集，因此不会有多个特征向量对应到同一个叶节点的情况。对于不在训练集中的叶节点，我们可以任意设置其类别，例如设置为训练集中出现次数最多的类别。这样，我们就得到了一个训练误差为0的决策树。

## 2.
### (1)
由于部分特征有较大误差，因此选择$w^TDw = \Sigma d_{ii}w_i^2$可以调节不同特征的权重，使误差较大的特征对结果的影响较小，而L2规范化无法调节权重。D的对角线元素可以影响各特征的权重，它的取值越大，该特征对结果的影响越小。
### (2)
$min_w (Xw-y)^2+\lambda w^TDw = min_w (Xw-y)^T(Xw-y)+\lambda w^TDw$
$$
\begin{align*}  
    \frac{\partial Loss+\lambda w^TDw}{\partial w} &= \frac{\partial}{\partial w}(Xw-y)^T(Xw-y)+\lambda w^TDw \\
    &= 2(Xw-y)^TX+\lambda(w^TD+w^TD^T) \\ 
    &= 2(Xw-y)^TX+2\lambda w^TD \\
    &= 0
\end{align*}
$$
即：
$$
\begin{align*}
    w^TX^TX+\lambda w^TD &= y^TX \\
    w^T(X^TX+\lambda D) &= y^TX \\
    w^T &= y^TX(X^TX+\lambda D)^{-1} \\
    w &= (X^TX+\lambda D)^{-1}X^Ty
\end{align*}
$$

## 3.
### (1)
显然$K_{i,j} = K(x_i, x_j) = K(x_j, x_i) = K_{j,i}$，因此是对称矩阵

### (2)
$$
\begin{align*}
    K &= (\phi(x_1), \phi(x_2), \cdots, \phi(x_n))^T(\phi(x_1), \phi(x_2), \cdots, \phi(x_n)) \\
    z^TKz &= z^T(\phi(x_1), \phi(x_2), \cdots, \phi(x_n))^T(\phi(x_1), \phi(x_2), \cdots, \phi(x_n))z \\
    &= ((\phi(x_1), \phi(x_2), \cdots, \phi(x_n))z)^T(\phi(x_1), \phi(x_2), \cdots, \phi(x_n))z \\
    &= (\sum_{i=1}^n z_i\phi(x_i))^T(\sum_{i=1}^n z_i\phi(x_i)) \\
    &= (\sum_{i=1}^n z_i\phi(x_i))^2 \\
    &\ge 0
\end{align*}
$$

## 4.


## 5.
$$
\begin{align*}
    \frac{\partial}{\partial w_j}L_{CE}(w,b) &= -\frac{\partial}{\partial w_j}[y\log\sigma(w^Tx+b)+(1-y)\log(1-\sigma(w^Tx+b))] \\
    &= -\frac{\partial}{\partial w_j}[y\log\sigma(\Sigma w_ix_i+b)+(1-y)\log(1-\sigma(\Sigma w_ix_i+b))] \\
    &= -y\frac{1}{\sigma(\Sigma w_ix_i+b)\ln2}\frac{\partial}{\partial w_j}\sigma(\Sigma w_ix_i+b)+(1-y)\frac{1}{(1-\sigma(\Sigma w_ix_i+b))\ln2}\frac{\partial}{\partial w_j}\sigma(\Sigma w_ix_i+b) \\
    &= x_j[(1-y)\frac{1}{(1-\sigma(w^Tx+b))\ln2}-y\frac{1}{\sigma(w^Tx+b)\ln2}]\sigma(w^Tx+b)(1-\sigma(w^Tx+b))
\end{align*}
$$

## 6.
一定会收敛。证明如下：
首先证明K-means算法的Los在迭代过程中是单调递减的。分别考虑K-means算法的两个步骤：
- 对于固定的聚类中心，优化聚类结果：$min_{C(1), \cdots, C(n)}Loss(\mu, C)$，此时由于取最小值，因此Loss不会增加。
- 对于固定的聚类结果，优化聚类中心：$min_{\mu(1), \cdots, \mu(k)}Loss(\mu, C)$，此时同样由于取最小值，因此Loss不会增加。

因此，K-means算法的Loss在迭代过程中是单调递减的，且Loss有下界（大于等于0），因此K-means算法一定会收敛。