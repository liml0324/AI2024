## 7.13
### a.
根据蕴含消去规则，$(P_1 \wedge ... \wedge P_m) \Rightarrow Q$等价于$\neg (P_1 \wedge ... \wedge P_m) \vee Q$，再根据德摩根律，$\neg (P_1 \wedge ... \wedge P_m) \vee Q$等价于$\neg P_1 \vee ... \vee \neg P_m \vee Q$。因此$(P_1 \wedge ... \wedge P_m) \Rightarrow Q$等价于$\neg P_1 \vee ... \vee \neg P_m \vee Q$。

### b.
根据析取的交换性，可以将任意一个子句通过交换文字顺序排列成$\neg P_1 \vee ... \vee \neg P_m \vee Q_1 \vee ... \vee Q_n$的形式（即前一部分全是否定文字，后一部分全是肯定文字）。根据蕴含消去规则，这个子句等价于$\neg(\neg P_1 \vee ... \vee \neg P_m) \Rightarrow (Q_1 \vee ... \vee Q_n)$，再根据德摩根律，$\neg(\neg P_1 \vee ... \vee \neg P_m) \Rightarrow (Q_1 \vee ... \vee Q_n)$等价于$(P_1 \wedge ... \wedge P_m) \Rightarrow (Q_1 \vee ... \vee Q_n)$。因此，任意一个子句都写成这种形式。

### c.
$$\frac{p_1 \wedge ... \wedge p_{n_p} \Rightarrow q_1 \vee ... \vee q_{n_q}, r_1 \wedge ... \wedge r_{n_r} \Rightarrow s_1 \vee ... \vee s_{n_s}}{p_1 \wedge ... \wedge p_{n_p} \wedge r_1 \wedge ... \wedge r_{j-1} \wedge r_{j+1} \wedge ... \wedge r_{n_r} \Rightarrow q_1 \vee ... \vee q_{i-1} \vee q_{i+1} \vee ... \vee q_{n_q} \vee s_1 \vee ... \vee s_{n_s}}$$
其中$q_i = r_j$。

## 证明前向链接算法的完备性
证明：
首先，前向链接算法每轮迭代会从队列中取出一个文字，而这个文字要么是知识库中原有的正文字（已知事实），要么是知识库中蕴含式的结论。而知识库中的子句数量有限，因此一定存在一个时间点，在这之后不会有新的结论被加入到已知事实中，将这个点称为不动点。

当达到不动点之后，将所有已知结论对应的文字赋值为true，其它文字赋值为false，即可得到KB的一个模型（设它为M）。这是因为，如果存在某个子句$a_1 \wedge ... \wedge a_n \Rightarrow p$在这个模型中为假，那么$a_1, ..., a_n$在已知事实集中，$p$不是已知事实，前向链接算法可以将$p$加入到已知事实中，与不动点的假设矛盾。

因为M是KB的一个模型，所以对任意的$q$，如果$KB \models q$，那么$q$在M中为真，即$q$在前向链接算法得到的已知事实中。因此，前向链接算法是完备的。