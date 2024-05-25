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
