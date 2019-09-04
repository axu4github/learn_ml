import random


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
            y = w * x ** 2 + b * x
            # y = w * x + b
            train_data.append((x, y))

        return train_data

    def update_param(self, param, lr, grad):
        return param - lr * grad

    def gradient_descent(self,
                         train_data,
                         w=random.random(),
                         b=random.random(),
                         ephocs=1000,
                         lr=0.0000001):
        """
        loss   => (w * x ** 2 + b * x - y) ** 2
        w_grad => 2.0 * (w * x ** 2 + b * x - y) * (x ** 2)
        b_grad => 2.0 * (w * x ** 2 + b * x - y) * (x)
        """
        print("Original Radom w & b => ({w}, {b})".format(w=w, b=b))
        for i in range(ephocs):
            for x, y in train_data:
                w_grad = 2.0 * (w * x ** 2 + b * x - y) * (x ** 2)
                b_grad = 2.0 * (w * x ** 2 + b * x - y) * (x)
                # w_grad = 2.0 * (w * x + b - y) * x
                # b_grad = 2.0 * (w * x + b - y)

                w = self.update_param(w, lr, w_grad)
                b = self.update_param(b, lr, b_grad)

        return (w, b)


if __name__ == "__main__":
    mgd = ManualGradientDescent()
    (w, b) = mgd.generate_w_b()
    print("Correct w & b => ({w}, {b})".format(w=w, b=b))
    train_data = mgd.generate_train_data(w, b, 1000)
    _w, _b = mgd.gradient_descent(train_data, ephocs=1000, lr=0.001)
    print("Predict w & b => ({w}, {b})".format(w=_w, b=_b))
