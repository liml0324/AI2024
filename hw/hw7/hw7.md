## 13.15
设事件A：（在不知道任何检测结果的情况下）我患有这种疾病
事件B：我在这种疾病的测试中得到阳性结果
则：
$P(A) = \frac{1}{10000}$
$P(B|A) = 0.99$
$P(B|\overline{A}) = 0.01$
$P(A|B) = \frac{P(B|A)P(A)}{P(B)} = \frac{P(B|A)P(A)}{P(B|A)P(A)+P(B|\overline{A})P(\overline{A})} = \frac{0.99\times\frac{1}{10000}}{0.99\times\frac{1}{10000}+0.01\times\frac{9999}{10000}} \simeq 0.0098$
因此，我确实患有这种疾病的概率为0.98%。

## 13.18
### a
设事件A：我取出的是伪币
事件B：我取出一枚硬币，抛出后正面朝上
$P(A) = \frac{1}{n}$
$P(B) = P(B|A)P(A)+P(B|\overline{A})P(\overline{A}) = \frac{1}{n}+\frac{1}{2}\times\frac{n-1}{n} = \frac{n+1}{2n}$
$P(A|B) = \frac{P(B|A)P(A)}{P(B)} = \frac{\frac{1}{n}}{\frac{n+1}{2n}} = \frac{2}{n+1}$

### b
抛出k次正面向上的概率为：$P(B_k) = P(B_k|A)P(A)+P(B_k|\overline{A})P(\overline{A}) = \frac{1}{n}+\frac{1}{2^k}\times\frac{n-1}{n} = \frac{n+2^k-1}{2^kn}$
$P(A|B_k) = \frac{P(B_k|A)P(A)}{P(B_k)} = \frac{\frac{1}{n}}{\frac{n+2^k-1}{2^kn}} = \frac{2^k}{n+2^k-1}$

### c
发生错误的概率为：
$P(\overline{A}B_k)+P(A\overline{B_k}) = P(\overline{A}|B_k)P(B_k)+P(\overline{B_k}|A)P(A) = \frac{n-1}{n+2^k-1}\times \frac{n+2^k-1}{2^kn} = \frac{n-1}{2^kn}$


## 13.21
### a
设事件A：肇事车看起来是蓝色的
事件B：肇事车是蓝色的
$P(B|A) = \frac{P(A|B)P(B)}{P(A)}$
$P(B)$的先验概率无法确定，因此无法计算$P(B|A)$

### b
如果已知雅典的出租车10辆中有9辆是绿色的，那么$P(B)=0.1$
$P(B|A) = \frac{P(A|B)P(B)}{P(A)} = \frac{0.75\times0.1}{P(A|B)P(B)+P(A|\overline{B})P(\overline{B})} = \frac{0.075}{0.075+0.25\times0.9} = 0.25$
因此最可能是绿色的。


## 13.22
### a
设一共有n种单词，代表第i种单词存在的事件为$E_i$。文档类别有m种，代表第j种类别的事件为$Q_j$。那么首先我们可以得到每种文档类别的先验概率$P(Q_j)$，即这类文档数量占总文档数量的比例；另外，我们可以用每种类别文档中每种单词出现的频率代表它在这类文档中出现的概率，即可以得到条件概率$P(E_i|Q_j),\ 1 \le i \le n,\ 1 \le j \le m$。

### b
我们可以测量出每种单词在该文档中是否出现。不妨设单词$E_1, ..., E_k$出现在该文档中，那么我们可以用贝叶斯定理计算出这个文档属于每种类别的概率$P(Q_j|E_1, ..., E_k)$，即：
$$
\begin{align}
P(Q_j|E_1, ..., E_k) &= \frac{P(E_1, ..., E_k|Q_j)P(Q_j)}{P(E_1, ..., E_k)}\\ 
&= \frac{P(Q_j)\Pi^k_{i=1} P(E_i|Q_j)}{\sum_{j=1}^m P(Q_j)\Pi^k_{i=1} P(E_i|Q_j)}
\end{align}
$$
上面这个计算式中，需要的所有的值都已知，因此可以计算出所有的$P(Q_j|E_1, ..., E_k)$，选择其中概率最大的类别作为这个文档的类别。
在计算时，由于分母对所有类别都是相同的，因此可以省略，只计算分子即可。

### c
我认为不太合理。单词的出现会受语法规则的限制和一些习惯表达的影响，因此它们的出现概率不是相互独立的。


## 14.12
### a
第2种和第3种是正确的。

### b
第2种最好。它在正确的同时，结构比第3种更简单。

### c
题中对产生“不超过1颗恒星的误差”的概率定义不明确，这里设多于1颗和少于1颗的概率均为e，即完全准确的概率为1-2e。
||M=0|1|2|3|4|
|:-:|:-:|:-:|:-:|:-:|:-:|
|N=1|e+f-ef|(1-2e)(1-f)|(1-f)e|0|0|
|2|f|(1-f)e|(1-2e)(1-f)|(1-f)e|0|
|3|f|0|(1-f)e|(1-2e)(1-f)|(1-f)e|

### d
考虑$M_1$，则N的可能取值有$M_1\plusmn 1\bigcup [M_1+3, +\infin)$
同样地，考虑$M_2$，则N的可能取值有$M_2\plusmn 1\bigcup [M_2+3, +\infin)$
实际可能的取值范围是两者的交集，即$\{2\} \bigcup \{4\} \bigcup [6, +\infin)$

### e
对于N=2的情况，我们可以计算：
$P(M_1=1, M_2=3|N=2) = e^2(1-f)^2$
然而对于N=4的情况，由于当星星数量为4个时，如果1号望远镜发生失焦，测量的结果有可能为1，也有可能为0，因此无法计算准确的概率，只能得到：
$P(M_1=1, M_2=3|N=4) \le fe(1-f)$
同理，对于$N\ge 6$的情况，我们也只能得到：
$P(M_1=1, M_2=3|N=6) \le f^2$
为求出最可能的星星数量，我们实际要计算的是$P(N=n|M_1=1, M_2=3) = \frac{P(M_1=1, M_2=3|N=n)P(N=n)}{P(M_1=1, M_2=3)}$。由于分母对每种情况都是相同的，只计算分子即可。
因此，要准确计算最可能的星星数量，我们必须知道$P(N)$的先验概率分布（即分子中的$P(N=n)$），且需要知道当发生失焦时，对所有的$k \ge 3$，测量得到的星星数量恰好为$N-k$的概率（用来准确计算$P(M_1=1, M_2=3|N=n)$）。

## 14.13
从枚举算法得到的计算式为：
$$
\begin{align*}
    P(N=n|M_1=2, M_2=2) &= \alpha\Sigma_{f_1}P(f_1)\Sigma_{f_2}P(f_2)P(N=n)P(M_1=2|f_1, n)P(M_2=2|f_2, n)
\end{align*}
$$
由于星星数量最多只有3颗，而失焦时测量结果至少会减少3颗，因此两台望远镜都没有失焦。
$$
\begin{align*}
    P(N=1|M_1=2, M_2=2) &= \alpha(1-f)^2e^2P(N=1) \\
    P(N=2|M_1=2, M_2=2) &= \alpha(1-f)^2(1-2e)^2P(N=2) \\
    P(N=3|M_1=2, M_2=2) &= \alpha(1-f)^2e^2P(N=3)
\end{align*}
$$
其中$\alpha$为归一化系数，$P(N=n)$为星星数量的先验概率分布。