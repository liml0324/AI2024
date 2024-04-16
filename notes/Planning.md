## 经典规划
### 集合论表达
状态表示为命题的集合，因此可能的状态有2^n种。
其上的一个动作会排除掉一些命题，而包含另一些命题。
目标状态是一组特定的命题，只要包含这些命题就是达到了目标状态。

### 经典表达
经典表达可以有有限多的谓词、变元、常元，但没有函数符号。一个状态由一组常元组成
谓词有状态谓词和刚性关系，前者是状态集的函数（如At(x,y)表示x在y处），后者是与状态无关的始终成立的关系。
规划操作是一个三元组 o = (name(o), precond(o), effects(o))，
其中：
- name(o)，操作的名字，形如 n(x1, . . . , xk)
- precond(o) 和 effects(o) 分别是 o 的前提和效果，都是文字集。刚性关系不能出现在任何操作 o 的效果中。action是一个操作用常元赋值的实例。
对经典表达进行grounding（即将所有变元用常元赋值）后与集合论表达等价，但结果可能指数增大（一个变元有n个常元，k个变元就有n^k个grounding）。

### 问题描述的计算复杂性
考虑两类问题：1. 给定一个规划问题，判断是否有解；2. 给定一个规划问题，判断是否存在长度小于等于n的解。
不允许函数符号时，这两个都是可判定的。如果允许函数符号，前者半可判定，后者可判定。
对于经典表达的经典规划，前者是EXPSPACE-complete（即至少与在指数空间中可解决的问题一样难），后者是NEXPTIME-complete。

### 状态空间规划
在状态转移图中搜索，有前向搜索（从初始状态开始），后向搜索（从目标状态开始），启发式搜索。
#### 前进规划
从初始状态开始，考虑所有可能的操作，直到找到一个目标状态。可以用深度优先搜索、广度优先搜索、A*搜索等。
#### 回归规划
从目标出发往回搜索
#### SPRIPS 规划


### 规划空间搜索
在行动序列组成的规划空间中搜索。尽管这个空间可能是无限的，但通常有较小的搜索空间。
具体搜索方法是，对于一个规划，找到它的一个缺陷，找到解决这个缺陷的所有方法，选择一个方法对规划进行求精。规划空间即使由规划作为节点，求精操作作为弧的一个图。
规划空间搜索采用最小承诺，即只承诺必要的条件（如，我们从目标状态往前搜索，为解决一个缺陷，加入一个在动作ac前的动作ap，则我们只承诺ap在ac前，而不承诺其他的次序关系）。

### 偏序规划
一个规划（行动序列$a_1, ..., a_n$）通常满足以下3点：
1. $a_1$在初始状态上可行
2. $a_{i-1}$后$a_i$可行
3. $a_n$执行后目标状态为真
通常的规划算法都是始终满足2，1、3中只有一个满足，通过搜索使另一个满足。而偏序规划保证1、3满足，2不满足，通过搜索使2满足。
偏序规划通过不断向上述的这样一个部分规划中添加因果链（即$a_{i-1}$的一个效果是Q，这个效果同时也是$a_i$的前提）来求解。
偏序规划也通过规划空间的搜索来实现（不满足偏序关系就是一种缺陷）。

## 新经典规划
### 规划图
规划图由两种节点组成：命题节点和动作节点。这两种节点组成的层交错排列，形成一个规划图。
搜索方法：先从初始状态开始构造规划图，不断扩展规划图直到目标状态的所有命题都在图中，然后从目标状态开始反向搜索，找到一个解。
扩展图的步骤是：首先将初始状态的所有命题加入P1层，然后将所有前提被满足（可采用）的行动构成A1层，根据A1层动作产生的效果构建P2层，如此循环，直到目标状态的所有命题都在图中。

构造过程要维护互斥关系：
- 两个行动 (或文字) 是互斥的，如果不存在可行规划同时包含两者。
- 两个行动是互斥的，如果:
  - 干扰 (Interference): 一个破坏另一个的效果或前提;
  - 需求竞争 (Competing needs): 两者的前提互斥。
- 两个命题是互斥的，如果同时获得两者所有方式都是互斥的。

规划解 (valid plan) 是规划图的一个子图，满足如下条件：
- 同一层的行动彼此不干扰;
- 在规划中每个行动的前提都被满足;
- 目标被满足。

### Planning as {SAT, CSP, ILP, …}
限制长度的规划问题是NP-complete，可以转换成SAT问题、CSP问题等问题进行求解。

## 经典规划的扩展