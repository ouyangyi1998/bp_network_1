# bp神经网络 iris数据集 不使用框架
- bp神经网络,不使用框架进行实现
- java 实现。。。
- bp神经网络模型包括输入层，隐含层，输出层 
- test.txt res.txt train.txt text用于测试 res输出结果 train训练 数据来自iris数据集
- 代码来自github @jingchenUSTC 感谢
- txt中的特征空间有四层 花卉种类有三种 通过dataNode中mAttribList(list)写入每一行的数据
- 通过train()方法对数据进行训练 先reset()初始化权值矩阵,前向传播forward(),反向传播backward(),权重更新updateWeight()
   - 把数据trainNode导入AnnClassifier 神经元 通过train方法导入每一行数据
   - 把test方法获得判断出来的属性
   - 最后输入res.txt 
- 在run configuration-program argument中设置值，在string[] args中获取 
- 建议 eta=0.02 time 5000 文件路径为相对路径
