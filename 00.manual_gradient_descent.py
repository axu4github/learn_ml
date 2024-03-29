import random
import logging as logger

logger.basicConfig(format="%(levelname)s %(message)s", level=logger.DEBUG)


class ManualGradientDescent(object):
    """
    手动实现梯度下降算法

    1. 随机参数生成一个目标函数，并记录参数 y = w * x ** 2 + b * x
    2. 随机创建数据
    3. 预设一个损失函数，并进行计算导数
    4. 进行梯度下降算法，并迭代数次，得到计算后的参数
    5. 对比参数是否和随机值接近
    """

    def __init__(self, *args, **kwargs):
        super(ManualGradientDescent, self).__init__(*args, **kwargs)

    def generate_w_b(self):
        """ 随机生成参数 """
        return (random.random(), random.random())

    def generate_train_data(self, w, b, total=1000):
        """ 生成训练数据 """
        train_data = []
        for i in range(total):
            x = random.random()
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
                         w=random.random(),
                         b=random.random(),
                         ephocs=1000,
                         lr=0.0000001):
        """
        梯度下降

        loss   => (w * x ** 2 + b * x - y) ** 2 # 定义一个损失函数
        w_grad => 2.0 * (w * x ** 2 + b * x - y) * (x ** 2) # 针对 w 求导
        b_grad => 2.0 * (w * x ** 2 + b * x - y) * (x) # 针对 b 求导
        """
        logger.info("Original Radom w & b => ({w}, {b})".format(w=w, b=b))
        for i in range(ephocs):
            for x, y in train_data:
                # 根据链式法则，针对参数对损失函数求导
                w_grad = 2.0 * (w * x ** 2 + b * x - y) * (x ** 2)
                b_grad = 2.0 * (w * x ** 2 + b * x - y) * (x)
                # w_grad = 2.0 * (w * x + b - y) * x
                # b_grad = 2.0 * (w * x + b - y)

                # 更新参数（优化，如何找到局部最优/全局最优?）
                w = self.update_param(w, lr, w_grad)
                b = self.update_param(b, lr, b_grad)

            # 输出损失
            _loss = self.loss(w * x ** 2 + b * x, y)
            logger.debug("Ephoc => {ephoc}, Loss => {loss}".format(
                ephoc=i + 1, loss=round(_loss, 10)))

        return (w, b)


if __name__ == "__main__":
    mgd = ManualGradientDescent()
    (w, b) = mgd.generate_w_b()
    logger.info("Correct w & b => ({w}, {b})".format(w=w, b=b))
    train_data = mgd.generate_train_data(w, b, 1000)
    _w, _b = mgd.gradient_descent(train_data, ephocs=5, lr=0.1)
    logger.info("Predict w & b => ({w}, {b})".format(w=_w, b=_b))

    """
    - TODO
    - 小批量随机梯度
    - 噪音
    - 激活函数
    - 优化函数
    """
