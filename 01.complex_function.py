import random
import logging as logger
import math
import matplotlib.pyplot as plt
import numpy as np

logger.basicConfig(format="%(levelname)s %(message)s", level=logger.INFO)


class ComplexFunction(object):
    """
    复杂函数
    模拟在不知道目标函数的情况下使用复杂函数进行测试，验证是否成功

    1. 随机参数生成一个目标函数，并记录参数 y = w * x ** 2 + b * x
    2. 随机创建数据
    3. 预设一个复杂函数（y = w3 * x ** 3 + w2 * x ** 2 + w1 * x + b）的损失函数，并进行计算导数
    4. 进行梯度下降算法，并迭代数次，得到计算后的参数
    5. 对比参数是否和随机值接近（w3=0, w2=w, w1=b, b=0）
    """

    def __init__(self, *args, **kwargs):
        super(ComplexFunction, self).__init__(*args, **kwargs)

    def generate_random_params(self):
        """ 随机生成参数 """
        return (random.random(), random.random())

    def generate_train_data(self, w, b, total=1000):
        """ 生成训练数据 """
        train_data, xs = [], []
        for i in range(total):
            xs.append(random.random())

        xs = sorted(xs)
        for x in xs:
            # 定义一个函数集合
            y = w * x ** 2 + b * x
            # y = w * x + b
            train_data.append((x, y))

        return train_data

    def update_param(self, param, lr, grad):
        return param - lr * grad

    def loss(self, yhat, y):
        """ 损失函数 """
        return (yhat - y) ** 2

    def gradient_descent(self,
                         train_data,
                         w3=random.random(),
                         w2=random.random(),
                         w1=random.random(),
                         b=random.random(),
                         ephocs=1000,
                         lr=0.0000001):
        """
        梯度下降

        loss   => (w * x ** 2 + b * x - y) ** 2 # 定义一个损失函数
        w_grad => 2.0 * (w * x ** 2 + b * x - y) * (x ** 2) # 针对 w 求导
        b_grad => 2.0 * (w * x ** 2 + b * x - y) * (x) # 针对 b 求导
        """
        logger.info("Original Radom w & b => ({w3}, {w2}, {w1}, {b})".format(
            w3=w3, w2=w2, w1=w1, b=b))
        for i in range(ephocs):
            # i = 0
            # while True:
            w3_lr, w2_lr, w1_lr, b_lr = 0, 0, 0, 0
            for x, y in train_data:
                # 根据链式法则，针对参数对损失函数求导
                w3_grad = 2.0 * (w2 * x ** 3 + w2 * x ** 2 +
                                 w1 * x + b - y) * (x ** 3)
                w2_grad = 2.0 * (w2 * x ** 3 + w2 * x ** 2 +
                                 w1 * x + b - y) * (x ** 2)
                w1_grad = 2.0 * (w2 * x ** 3 + w2 * x **
                                 2 + w1 * x + b - y) * (x)
                b_grad = 2.0 * (w2 * x ** 3 + w2 * x ** 2 + w1 * x + b - y)
                # w_grad = 2.0 * (w * x + b - y) * x
                # b_grad = 2.0 * (w * x + b - y)
                #
                w3_lr = w3_lr + w3_grad ** 2
                w2_lr = w2_lr + w2_grad ** 2
                w1_lr = w1_lr + w1_grad ** 2
                b_lr = b_lr + b_grad ** 2

                # 更新参数（优化，如何找到局部最优/全局最优?）
                w3 = self.update_param(w3, lr / math.sqrt(w3_lr), w3_grad)
                w2 = self.update_param(w2, lr / math.sqrt(w2_lr), w2_grad)
                w1 = self.update_param(w1, lr / math.sqrt(w1_lr), w1_grad)
                b = self.update_param(b, lr / math.sqrt(b_lr), b_grad)

            # 输出损失
            _loss = self.loss(w3 * x ** 3 + w2 * x ** 2 + w1 * x + b, y)
            logger.info("Ephoc => {ephoc}, Loss => {loss}, params => ({w3}, {w2}, {w1}, {b})".format(
                ephoc=i + 1, loss=round(_loss, 10), w3=w3, w2=w2, w1=w1, b=b))

            # if _loss < 0.000001:
            #     break

            # i += 1

        return (w3, w2, w1, b)


if __name__ == "__main__":
    cf = ComplexFunction()
    (w, b) = cf.generate_random_params()
    for i in range(10):
        train_data = cf.generate_train_data(w, b, 100000)
        print(train_data[0:10])
        w3, w2, w1, _b = cf.gradient_descent(train_data, ephocs=1, lr=0.000001)
        logger.info("Correct w & b => ({w}, {b})".format(w=w, b=b))
        logger.info("Predict w3, w2, w1, b => ({w3}, {w2}, {w1}, {b})".format(
            w3=w3, w2=w2, w1=w1, b=_b))

        x = np.linspace(-1000, 1000)
        y = w3 * x ** 3 + w2 * x ** 2 + w1 * x + b

        plt.plot(x, y)
    plt.show()
    """
    - TODO
    - 小批量随机梯度
    - 噪音
    - 激活函数
    - 优化函数
    """
