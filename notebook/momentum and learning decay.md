# 0. 动量与学习衰减
1. https://www.bilibili.com/video/BV18g4119737/?p=58&vd_source=70200f7d09862fd682e5f89b22c89125
2. Regularization里讲了“weight decay”是迫使weight的范数接近0，这里learning rate的decay也是逐渐减小learning rate
3. Momentum动量是“刹不住车”

# 1. 动量
## 1.1 为什么要考虑动量
1. ![](https://gitee.com/wyjyoga/my-pic-go/raw/master/img/20221215201740.png)
    - $z^{k+1}$ 是历史方向$z^{k}$和梯度方向的一个结合。如果比较看重历史方向 aka 惯性比较大 aka $\beta$比较大，那么在梯度更新的时候就会更加偏向历史方向（可以根据平行四边形法则描绘方向）
2. 不考虑动量可能出现的问题：
    ![](https://gitee.com/wyjyoga/my-pic-go/raw/master/img/20221215202301.png)
    - 不考虑历史会使得**初始学习阶段非常震荡**

3. 考虑动量之后
    ![](https://gitee.com/wyjyoga/my-pic-go/raw/master/img/20221215202611.png)

## 1.2 pytorch的动量设置
1. 一些比如`SGD`的优化器需要手动设置动量：
![](https://gitee.com/wyjyoga/my-pic-go/raw/master/img/20221215202800.png)
2. 但是比如`Adam`已经包括了动量，所以无须设置这个系数

# 2. 学习率
## 2.1 学习率的影响
1.太小：学的慢，太大：震荡
2. 设置了lr decay的结果：
![](https://gitee.com/wyjyoga/my-pic-go/raw/master/img/20221215203313.png)

## 2.2 pytorch 与lr decay
1. 一个decay的方法：`scheduler = ReduceLROnPlateau(optimizer，min)`
    - 意思是到plateau（loss不动了呈现出一个“平原”）时，进行衰退
    - 一般在train完,本轮的梯度更新完之后，进行`scheduler.step(loss_val)`
2. 另一个监听方法，简单粗暴：`scheduler=StepLR(optimizer,step_size=30,gamma=0.1)`
    - 也是使用`.step`方法进行lr的更新
    - `step_size`和`gamma`的意思是每进行30个epoch，lr会*0.1
    - 