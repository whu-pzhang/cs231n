# CS231n fall 2018 assignment

## Assignment1

### kNN

1. `np.argsort()`

可对数组排序，默认返回从小到大的元素索引

2. `np.bincount()`

将数组元素当做索引展开，与`np.argmax()` 配合可以得到数组中出现次数最多的元素

3. 测试集与训练集的$L2$范数计算的向量化

训练集 $P \in R^{m \times d}$， 测试集 $Q \in R^{n \times d}$，需计算每个测试样本与训练集中
所有样本的欧几里得距离。

需要先对范数进行展开，然后利用广播和矩阵乘法来完成。见[这三行代码](https://github.com/whu-pzhang/cs231n/blob/master/assignment1/cs231n/classifiers/k_nearest_neighbor.py#L126-L128)
