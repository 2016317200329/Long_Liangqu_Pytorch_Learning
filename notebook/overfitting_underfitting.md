# 0. 过拟合和欠拟合
https://www.bilibili.com/video/BV18g4119737?p=54&vd_source=70200f7d09862fd682e5f89b22c89125


# 1. 模型能力与数据学习
1. 对于一些样本，我们会选择不同类型的模型去学习
    - 以多项式模型来说，不同的次数，对应着不同的“抖动性”
    - ![](https://gitee.com/wyjyoga/my-pic-go/raw/master/img/20221208171850.png)
2. 如何衡量一个模型的**学习能力/ model capacity** ？
    - 多项式模型的次数越高，表达能力越强
    - 网络越深，参数越多，表达能力越强


# 2. underfitting
1. 如果模型的能力小于数据的ground-truth pattern，则欠拟合
2. 表现在：
    - train acc is bad or loss is bad
    - test acc is bad or loss is bad
    - 不好检测actually

# 3. overfitting
1. 表现在：
   - train loss and acc. is much better but test acc is worse
2. overfitting强，generalization performance就会差
3. 现在普遍overfitting