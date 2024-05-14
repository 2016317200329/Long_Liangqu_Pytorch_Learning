# 0.
1. https://www.bilibili.com/video/BV18g4119737?p=61&vd_source=70200f7d09862fd682e5f89b22c89125
2. 什么是卷积

# 1. 什么是卷积
## 1.1 为什么要叫“卷积”
1. “偏移运算”
   ![](https://gitee.com/wyjyoga/my-pic-go/raw/master/img/20230107145326.png)
   ![](https://gitee.com/wyjyoga/my-pic-go/raw/master/img/20230107145449.png)

# 2. 不同的卷积核
1. 不同的卷积核有不同的效果：
![](https://gitee.com/wyjyoga/my-pic-go/raw/master/img/20230107145854.png)
2. 以及
![](https://gitee.com/wyjyoga/my-pic-go/raw/master/img/20230107150103.png)

# 3. conv net
1. 用什么样的kernel决定了从什么角度看图片，kernel是一个moving window
2. ![](https://gitee.com/wyjyoga/my-pic-go/raw/master/img/20230107150451.png)

## 3.1 multi-kernel
1. 一定要分清楚：![](https://gitee.com/wyjyoga/my-pic-go/raw/master/img/20230107151617.png)
   - multi-k的4维数据里，第一个16指的是kernel_num，也就是我打算用16个角度观察这张图片
   - 一个kernel只有一个bias，因此这里是16
   - out的维度里，kernel_num变成了out的chanel值，batch_size的那个b不变，output最后2维是28跟步长和padding也有关系
## 3.2 LeNet-5
1. 
   