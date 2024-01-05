# 1. 时间序列表示方法

1. 以一个语言序列为例，一句话包含5个单词，每个单词有自己的embedding（长度为100的话）
   
   - 这sequence的shape就是(5,100)
   
   - 引入batch后，会变成(b,5,100)或者(5,b,100)

2. 对于RNN，一般一次会input**一帧进去**，也就是一个单词的意思，
   
   - 所以一次input的大小一般写作是(5,b,100)

# 2. RNN原理

## 2.1 基本model

1. Naive version
   
   <img title="" src="file:///C:/Users/Wang%20Yujia/AppData/Roaming/marktext/images/2022-10-03-16-41-21-image.png" alt="" width="590">
   
   - 这个5个frame下，每个frame对应的处理单元的权值都不同。当句子加长的时候，会发现需要的units变多

2. weight sharing：
   
   <img src="file:///C:/Users/Wang%20Yujia/AppData/Roaming/marktext/images/2022-10-03-16-42-35-image.png" title="" alt="" width="518">
   
   - 在这之后所有的w和b都一样，都用一个unit处理，意味着这个线性层要有能力处理所有单词
   
   - 这样对长句子也不会增加参数

3. with consistent memory
   
   <img src="file:///C:/Users/Wang%20Yujia/AppData/Roaming/marktext/images/2022-10-03-16-44-48-image.png" title="" alt="" width="696">
   
   - h表示memory单元，在这个case中，$h_0$为全0vec。
   
   - h的大小都和一个frame处理的数据大小一样的：[3,100]

4. In math
   
   <img src="file:///C:/Users/Wang%20Yujia/AppData/Roaming/marktext/images/2022-10-03-16-47-03-image.png" title="" alt="" width="361">
   
   - $h_t$来自于两个部分：$h_{t-1}$和$x_t$，两个W分别对$h_{t-1}$和$x_t$做**特征提取（也就是矩阵相乘）**
   
   - $h_t$的激活函数用的是$tanh$
   
   - $y_t$可以理解为linear层

## 2.3 梯度推导

见ipad，问题在于梯度里会有一个$W_hh$的连乘

# 3. RNN Layer

1. <img src="file:///C:/Users/Wang%20Yujia/AppData/Roaming/marktext/images/2022-10-03-18-56-04-image.png" title="" alt="" width="630">
## 3.2 多层的输出是什么
1. `h`是**最后一个timestamp上所有memeory**的状态，`out`是**最后一个timestamp上最后一个memory**的输出
![](https://gitee.com/wyjyoga/my-pic-go/raw/master/img/20230706094321.png)

2. 