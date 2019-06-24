\---

layout:     post

title:      "DecisionTree"

subtitle:   " \"机器学习笔记--常见算法(2)--决策树算法介绍\""

date:       2019-06-24 15:30:00

author:     "voidwalker"

header-img: "img/post-bg-2015.jpg"

catalog: true

tags:

\- 机器学习



\---





[TOC]



## 1.决策树简介 

决策树：既能做分类，又能做回归 

决策树模型是一种传统的算法，决策树实际上就是在模仿人类做决策的过程。 



可以从两个方面来理解决策树: 

（1）Aggregation model 

Aggregation model：aggregation的核心就是将许多可供选择使用的比较好的hypothesis融合起来，利用集体的智慧组合成G，使其得到更好的机器学习预测模型。 

决策树的整个流程类似一个树状结构。如图：

![image-20190623143827680](/Users/voidwalker/Library/Application Support/typora-user-images/image-20190623143827680.png)

把这种树状结构对应到一个hypothesis G(x)中，G(x)的表达式为： 

$$G(x)=\sum_{t=1}^Tq_t(x)\cdot g_t(x)$$

G(x)由许多$g_t(x)$组成，即aggregation的做法。每个$g_t(x)$就代表上图中的蓝色圆圈（树的叶子）。这里的$g_t(x)$是常数，因为是处理简单的classification问题。我们把这些$g_t(x)$称为base hypothesis。$q_t(x)$表示每个$g_t(x)$成立的条件，代表上图中橘色箭头的部分。不同的$g_t(x)$对应于不同的$q_t(x)$，即从树的根部到顶端叶子的路径不同。图中中的菱形代表每个简单的节点。所以，这些base hypothesis和conditions就构成了整个G(x)的形式，就像一棵树一样，从根部到顶端所有的叶子都安全映射到上述公式上去了。



（2）条件分支的思想 

将整体G(x)分成若干个$G_c(x)$，也就是把整个大树分成若干个小树，如下所示：

$$G(x)=\sum_{c=1}^C[b(x)=c]\cdot G_c(x)$$

上式中，G(x)表示完整的大树，即full-tree hypothesis，b(x)表示每个分支条件，即branching criteria，$G_c(x)$表示第c个分支下的子树，即sub-tree。这种结构被称为递归型的数据结构，即将大树分割成不同的小树，再将小树继续分割成更小的子树。所以，决策树可以分为两部分：root和sub-trees。



**决策树分成4个部分： **

根结点（数据集D） 

分支（属性测试划分） 

非叶子结点（决策点） 

叶子结点（决策结果） 



**决策树评价：优缺点**

决策树的优点： 

模型简单、便于理解、应用广泛 

算法简单，容易实现 

训练和预测时，效率较高 

决策树缺点： 

缺少足够的理论支持 

如何选择合适的树结构对初学者来说比较困惑 

决策树代表性的演算法比较少 



## 2.决策树Decision Tree

我们可以用递归形式将decision tree表示出来，它的基本的算法可以写成：

![这里写图片描述](http://img.blog.csdn.net/20170720073120706?)

这个Basic Decision Tree Algorithm的流程可以分成四个部分，首先学习设定划分不同分支的标准和条件是什么；接着将整体数据集D根据分支个数C和条件，划为不同分支下的子集Dc；然后对每个分支下的Dc进行训练，得到相应的机器学习模型Gc；最后将所有分支下的Gc合并到一起，组成大矩G(x)。但值得注意的是，这种递归的形式需要终止条件，否则程序将一直进行下去。当满足递归的终止条件之后，将会返回基本的hypothesis $g_t(x)$。

![这里写图片描述](http://img.blog.csdn.net/20170720074057776?)

所以，决策树的基本演算法包含了四个选择：

- **分支个数（number of branches）（$c$）**

- **分支条件（branching criteria）（$b(x)$）**

- **终止条件（termination criteria）**

- **基本算法（base hypothesis）**($g_t(x)$)

  





## 3.特征选择

（特征选择对应上面提到的分支条件$b(x)$）

特征选择在于选取对训练数据具有分类能力的特征。

特征选择是决定用哪个特征来划分特征空间。问题是：究竟选择哪个特征更好些？使用纯净度purifying这个概念来选择最好的decision stump. purifying的核心思想就是每次切割都尽可能让左子树和右子树中同类样本占得比例最大或者Yn都很接近( regression) ,即错误率最小。比如说classifiacation问题中 ,如果左子树全是正样本,右子树全是负样本,那么它的纯净度就很大,说明该分支效果很好。

确定选择特征的准则。信息增益、信息增益率、Gini系数能够很好地表示这些准则。

### 3.1 信息增益

**信息熵(information entropy) H(X)**

信息熵是度量样本集合纯度最常用的一种指标。熵的值越小，则样本集合$D$的纯度越高。

### 设X是一个取有限个值的离散随机变量，其概率分布为：

![image-20190623151351702](/Users/voidwalker/Library/Application Support/typora-user-images/image-20190623151351702.png)

随机变量X的熵定义为：

![image-20190623151149459](/Users/voidwalker/Library/Application Support/typora-user-images/image-20190623151149459.png)



**条件熵(conditional entropy) H(Y|X)**

条件熵H(Y|X)表示在已知随机变量X的条件下随机变量Y的不确定性。定义为：

![image-20190623151537040](/Users/voidwalker/Library/Application Support/typora-user-images/image-20190623151537040.png)

这里，$p_i=P(X=x_i), i = 1,2,…,n$



**信息增益(information gain)**

信息增益表示得知特征X的信息而使得类Y的信息的不确定性减少的程度。

特征A对训练数据集D的信息增益g(D,A)，定义为：

![image-20190623151852472](/Users/voidwalker/Library/Application Support/typora-user-images/image-20190623151852472.png)



**信息增益的算法**

![image-20190623151925886](/Users/voidwalker/Library/Application Support/typora-user-images/image-20190623151925886.png)

### 3.2 信息增益比

信息增益对可取值数目较多的属性有所偏好，为减少这种偏好可能带来的不利影响，引入信息增益率来选择最优化分属性。

**信息增益比(information gain ratio) **

特征A对训练数据集D的信息增益比定义为：

![image-20190623211530618](/Users/voidwalker/Library/Application Support/typora-user-images/image-20190623211530618.png)

其中，

![image-20190623211547949](/Users/voidwalker/Library/Application Support/typora-user-images/image-20190623211547949.png)

### 3.3 Gini系数

分类问题中，假设有k个类，样本点属于第k类的概率为$p_k$，则概率分布的基尼指数定义为：

![image-20190623150143332](/Users/voidwalker/Library/Application Support/typora-user-images/image-20190623150143332.png)

如果样本集合D根据特征A是否取某一可能值a被分割成$D_1$和$D_2$两部分，即

![image-20190623152721479](/Users/voidwalker/Library/Application Support/typora-user-images/image-20190623152721479.png)

则在特征A的条件下，集合D的基尼指数定义为

![image-20190623152736346](/Users/voidwalker/Library/Application Support/typora-user-images/image-20190623152736346.png)



## 4.决策树的生成 

ID3算法、C4.5算法、CART算法



ID3：由特征选择和树的生成组成，无剪枝，所以该算法生成的树容易产生过拟合。 

C4.5：由特征选择，树的生成，剪枝组成。 

CART：由特征选择，树的生成，剪枝组成。既可用于分类也可用于回归。CART假设决策树是二叉树。 



前面提到：决策树的基本演算法包括下面四个部分：

- **分支个数（number of branches）（$c$）**

- **分支条件（branching criteria）（$b(x)$）**

- **终止条件（termination criteria）**

- **基本算法（base hypothesis）**($g_t(x)$)

  

**分支个数c:**

在ID3和C4.5中，一个结点的分支个数为以某个特征作为划分时，该特征所有可取值的个数。

在CART中，人为规定分支个数为2(即c=2)。

**分支条件b(x)：对应3.0特征选择**

ID3：信息增益

C4.5：信息增益率

CART：Gini系数(分类)/平方误差准则(回归)

**终止条件：**

决策树的生成是一个递归过程.在决策树基本算法中,有三种情形会导致递归返回: (1) 当前结点包含的样本全属于同一类别,无需划分; (2)当前属性集为空，或是所有样本在所有属性上取值相同,无法划分; (3)当前结点包含的样本集合为空,不能划分.

**基本算法($g_t(x)$)：**

($g_t(x)$)：每个分支最后的$g_t(x)$（数的叶子）是一个常数。按照最小化$E_{in}$的目标，对于binary/multiclass classification(0/1 error)问题，看正类和负类哪个更多，$g_t(x)$取所占比例最多的那一类$y_n$；对于regression(squared error)问题，$g_t(x)$则取所有$y_n$的平均值。





## 5.决策树的剪枝

剪枝(pruning)是决策树学习算法对付“过拟合”的主要手段.在决策树学习中，为了尽可能正确分类训练样本,结点划分过程将不断重复,有时会造成决策树分支过多,这时就可能因训练样本学得“太好”了，以致于把训练集自身的一些特点当作所有数据都具有的一般性质而导致过拟合.因此,可通过主动去掉一些分支来降低过拟合的风险.

(把树切的太碎，决策树容易学到一些噪音点，在真正测试时实际效果不好。出现过拟合问题)



预剪枝:在构建决策树的过程时，提前停止。（如进行高度的限制） 

后剪枝:决策树构建好后，然后才开始裁剪。 

**决策树剪枝分为“预剪枝”(prepruning)和“后剪枝”(post-pruning) 两种。**



预剪枝:在构建决策树的过程时，提前停止。（如进行高度的限制） 

后剪枝:决策树构建好后，然后才开始裁剪。 自底相上地对非叶结点进行考察,若将该结点对应的子树替换为叶结点能带来决策树泛化性能提升,则将该子树替换为叶结点。



**剪枝算法介绍**

决策树的剪枝往往通过极小化决策树整体的损失函数或代价函数来实现。

**损失函数**

设树T的叶节点个数为|T|，t是树T的叶结点，该叶结点有$N_t$个样本点，其中k类的样本点有$Ntk$个，k=1,2,…,K, $H_t(T)$为叶结点t上的经验熵，a>=0为参数，则决策树学习的损失函数可以定义为：

![image-20190623162951676](/Users/voidwalker/Library/Application Support/typora-user-images/image-20190623162951676.png)

其中经验熵为

![image-20190623163022193](/Users/voidwalker/Library/Application Support/typora-user-images/image-20190623163022193.png)

在损失函数中，将![image-20190623163112276](/Users/voidwalker/Library/Application Support/typora-user-images/image-20190623163112276.png)第1项记作：

![image-20190623163053012](/Users/voidwalker/Library/Application Support/typora-user-images/image-20190623163053012.png)

这时有，

![image-20190623163153553](/Users/voidwalker/Library/Application Support/typora-user-images/image-20190623163153553.png)

C(T)表示模型对训练数据的预测误差，即模型与训练数据的拟合程度，|T|表示模型复杂度，参数a≥0控制两者之间的影响。较大的a促使选择较简单的模型(树),较小的a促使选择较复杂的模型(树)。a= 0意味着只考虑模型与训练数据的拟合程度,不考虑模型的复杂度。

剪枝,就是当a确定时,选择损失函数最小的模型, 即损失函数最小的子树。当a值确定时,子树越大,往往与训练数据的拟合越好,但是模型的复杂度就越高;相反,子树越小,模型的复杂度就越低，但是往往与训练数据的拟合不好。损失函数正好表示了对两者的平衡。

![image-20190623163411374](/Users/voidwalker/Library/Application Support/typora-user-images/image-20190623163411374.png)



**决策树剪枝评价：预剪枝和后剪枝的优缺点**

预剪枝使得决策树的很多分支都没有“展开”，这不仅降低了过拟合的风险,还显著减少了决策树的训练时间开销和测试时间开销.但另一方面,有些分支的当前划分虽不能提升泛化性能、甚至可能导致泛化性能暂时下降,但在其基础上进行的后续划分却有可能导致性能显著提高;预剪枝基于“贪心”本质禁止这些分支展开，给预剪枝决策树带来了欠拟合的风险.

后剪枝决策树通常比预剪枝决策树保留了更多的分支. 一般情形下，后剪枝决策树的欠拟合风险很小，泛化性能往往优于预剪枝决策树.但后剪枝过程是在生成完全决策树之后进行的，并且要自底向上地对树中的所有非叶结点进行逐一考察,因此其训练时间开销比未剪枝决策树和预剪枝决策树都要大得多，



## 6.连续与缺失值 

### 6.1 连续值处理

连续属性离散化技术。最简单的策略是采用二分法对连续属性进行处理(C4.5)。

给定样本集D和连续属性a,假定a在D上出现了n个不同的取值,将这些值从小到大进行排序,记为${a^1,a^2,...,a^n}$.基于划分点t可将D分为子集$D_t^-$和$D_t^+$,其中$D_t^-$包含那些在属性a上取值不大于t的样本,而$D_t^+$则包含那些在属性a上取值大于t的样本.

注：对连续属性a，我们可考察包含n-1个元素的候选划分点结合，即把区间$[a^i,a^{i+1})$的中位点$\frac{[a^i,a^{i+1})}{2}$作为候选划分点

![image-20190623220107737](/Users/voidwalker/Library/Application Support/typora-user-images/image-20190623220107737.png)



### 6.2 缺失值处理

我们需解决两个问题:

 (1)如何在属性值缺失的情况下进行划分属性选择? 

(2)给定划分属性,若样本在该属性上的值缺失，如何对样本进行划分?

(1)划分属性选择：

我们仅可根据D在属性a上没有缺失值的样本子集来判断属性a的优劣。如有17个样例，14个样例无缺失值，则根据这14个值来判断属性a的优劣。

(2)对样本进行划分：

若样本x在划分属性a上的取值已知，则将x划入与其属性对应的子结点，且样本权重在子结点保持为$\omega_x$。

若样本x在划分属性a上的取值未知，则将x同时划入所有子结点，且样本权值在与属性值$a^v$对应的子结点中调整为$\widetilde{r}_v\cdot\omega_x$。直观地看，这就是让同一个样本以不同的概率划入到不同的子结点中去。

公式及例子详见周志华《机器学习》P85



特征缺失（台大《机器学习技法》课程）： 

surrogate branch，即寻找与该特征相似的替代feature。如何确定相似的feature？做法是在决策树训练的时候，找出与该特征相似的feature，如果替代的feature与原feature切割的方式和结果是类似的，那么就表明二者是相似的，就把该替代的feature也存储下来。当预测时遇到原feature缺失的情况，就用替代feature进行分支判断和选择。 