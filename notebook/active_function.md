# 0. 常用的激活函数以及GPU运算
1. https://www.bilibili.com/video/BV18g4119737?p=51&vd_source=70200f7d09862fd682e5f89b22c89125
2. 只记录几个不常见的

# 1. 激活函数：Relu的变种
1. leaky relu：避免了当x<=0的时候梯度一直是0
![pic](https://gitee.com/wyjyoga/my-pic-go/raw/master/img/20221208151528.png)
2. SELU: 比Relu更光滑,公式前半段是relu
![pic](https://gitee.com/wyjyoga/my-pic-go/raw/master/img/20221208151703.png)
3. softplus：在x=0处做均匀处理
![pic]((https://gitee.com/wyjyoga/my-pic-go/raw/master/img/20221208151814.png))
