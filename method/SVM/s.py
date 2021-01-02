import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler


class SMOStruct:
    """ 按照John Platt的论文构造SMO的数据结构"""

    def __init__(self, X, y, C, kernel, alphas, b, errors, user_linear_optim):
        self.X = X  # 训练样本
        self.y = y  # 类别 label
        self.C = C  # regularization parameter  正则化常量，用于调整（过）拟合的程度
        self.kernel = kernel  # kernel function   核函数，实现了两个核函数，线性和高斯（RBF）
        self.alphas = alphas  # lagrange multiplier 拉格朗日乘子，与样本一一相对
        self.b = b  # scalar bias term 标量，偏移量
        self.errors = errors  # error cache  用于存储alpha值实际与预测值得差值，与样本数量一一相对

        self.m, self.n = np.shape(
            self.X)  # store size(m) of training set and the number of features(n) for each example
        # 训练样本的个数和每个样本的features数量

        self.user_linear_optim = user_linear_optim  # 判断模型是否使用线性核函数
        self.w = np.zeros(self.n)  # 初始化权重w的值，主要用于线性核函数
        # self.b = 0


def linear_kernel(x, y, b=1):
    # 线性核函数
    """ returns the linear combination of arrays 'x' and 'y' with
    the optional bias term 'b' (set to 1 by default). """
    result = x @ y.T + b
    return result  # Note the @ operator for matrix multiplications


# 判别函数一，用于单一样本
def decision_function_output(model, i):
    if model.user_linear_optim:
        # Equation (J1)
        # return float(np.dot(model.w.T, model.X[i])) - model.b
        return float(model.w.T @ model.X[i]) - model.b
    else:
        # Equation (J10)
        return np.sum(
            [model.alphas[j] * model.y[j] * model.kernel(model.X[j], model.X[i]) for j in range(model.m)]) - model.b


# 判别函数二，用于多个样本
def decision_function(alphas, target, kernel, X_train, x_test, b):
    """ Applies the SVM decision function to the input feature vectors in 'x_test'.
    """
    result = (alphas * target) @ kernel(X_train, x_test) - b  # *，@ 两个Operators的区别?

    return result


def plot_decision_boundary(model):
    plt.scatter(model.X[:, 0], model.X[:, 1], c=model.y, lw=0, alpha=0.25)
    x1 = model.X[:, 0].max()
    x2 = model.X[:, 0].min()
    a1, a2 = model.w
    b = float(model.b)
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    plt.show()



# 选择了alpha2, alpha1后开始第一步优化，然后迭代， “第二层循环，内循环”
# 主要的优化步骤在这里发生
def take_step(i1, i2, model):
    # skip if chosen alphas are the same
    if i1 == i2:
        return 0, model
    # a1, a2 的原值，old value，优化在于产生优化后的值，新值 new value
    # 如下的alph1,2, y1,y2,E1, E2, s 都是论文中出现的变量，含义与论文一致
    alph1 = model.alphas[i1]
    alph2 = model.alphas[i2]

    y1 = model.y[i1]
    y2 = model.y[i2]

    E1 = get_error(model, i1)
    E2 = get_error(model, i2)
    s = y1 * y2

    # 计算alpha的边界，L, H
    # compute L & H, the bounds on new possible alpha values
    if (y1 != y2):
        # y1,y2 异号，使用 Equation (J13)
        L = max(0, alph2 - alph1)
        H = min(model.C, model.C + alph2 - alph1)
    elif (y1 == y2):
        # y1,y2 同号，使用 Equation (J14)
        L = max(0, alph1 + alph2 - model.C)
        H = min(model.C, alph1 + alph2)
    if (L == H):
        return 0, model

    # 分别计算啊样本1, 2对应的核函数组合，目的在于计算eta
    # 也就是求一阶导数后的值，目的在于计算a2new
    k11 = model.kernel(model.X[i1], model.X[i1])
    k12 = model.kernel(model.X[i1], model.X[i2])
    k22 = model.kernel(model.X[i2], model.X[i2])
    # 计算 eta，equation (J15)
    eta = k11 + k22 - 2 * k12

    # 如论文中所述，分两种情况根据eta为正positive 还是为负或0来计算计算a2 new

    if (eta > 0):
        # equation (J16) 计算alpha2
        a2 = alph2 + y2 * (E1 - E2) / eta
        # clip a2 based on bounds L & H
        # 把a2夹到限定区间 equation （J17）
        if L < a2 < H:
            a2 = a2
        elif (a2 <= L):
            a2 = L
        elif (a2 >= H):
            a2 = H
    # 如果eta不为正（为负或0）
    # if eta is non-positive, move new a2 to bound with greater objective function value
    else:
        # Equation （J19）
        # 在特殊情况下，eta可能不为正not be positive
        f1 = y1 * (E1 + model.b) - alph1 * k11 - s * alph2 * k12
        f2 = y2 * (E2 + model.b) - s * alph1 * k12 - alph2 * k22

        L1 = alph1 + s * (alph2 - L)
        H1 = alph1 + s * (alph2 - H)

        Lobj = L1 * f1 + L * f2 + 0.5 * (L1 ** 2) * k11 \
               + 0.5 * (L ** 2) * k22 + s * L * L1 * k12

        Hobj = H1 * f1 + H * f2 + 0.5 * (H1 ** 2) * k11 \
               + 0.5 * (H ** 2) * k22 + s * H * H1 * k12

        if Lobj < Hobj - eps:
            a2 = L
        elif Lobj > Hobj + eps:
            a2 = H
        else:
            a2 = alph2

    # 当new a2 千万分之一接近C或0是，就让它等于C或0
    if a2 < 1e-8:
        a2 = 0.0
    elif a2 > (model.C - 1e-8):
        a2 = model.C
    # 超过容差仍不能优化时，跳过
    # If examples can't be optimized within epsilon(eps), skip this pair
    if (np.abs(a2 - alph2) < eps * (a2 + alph2 + eps)):
        return 0, model

    # 根据新 a2计算 新 a1 Equation(J18)
    a1 = alph1 + s * (alph2 - a2)

    # 更新 bias b的值 Equation (J20)
    b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + model.b
    # equation (J21)
    b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + model.b

    # Set new threshoold based on if a1 or a2 is bound by L and/or H
    if 0 < a1 and a1 < C:
        b_new = b1
    elif 0 < a2 and a2 < C:
        b_new = b2
    # Average thresholds if both are bound
    else:
        b_new = (b1 + b2) * 0.5

    # update model threshold
    model.b = b_new

    # 当所训练模型为线性核函数时
    # Equation (J22) 计算w的值
    if model.user_linear_optim:
        model.w = model.w + y1 * (a1 - alph1) * model.X[i1] + y2 * (a2 - alph2) * model.X[i2]
    # 在alphas矩阵中分别更新a1, a2的值
    # Update model object with new alphas & threshold
    model.alphas[i1] = a1
    model.alphas[i2] = a2

    # 优化完了，更新差值矩阵的对应值
    # 同时更新差值矩阵其它值
    model.errors[i1] = 0
    model.errors[i2] = 0
    # 更新差值 Equation (12)
    for i in range(model.m):
        if 0 < model.alphas[i] < model.C:
            model.errors[i] += y1 * (a1 - alph1) * model.kernel(model.X[i1], model.X[i]) + \
                               y2 * (a2 - alph2) * model.kernel(model.X[i2], model.X[i]) + model.b - b_new

    return 1, model


def get_error(model, i1):
    if 0 < model.alphas[i1] < model.C:
        return model.errors[i1]
    else:
        return decision_function_output(model, i1) - model.y[i1]


def examine_example(i2, model):
    y2 = model.y[i2]
    alph2 = model.alphas[i2]
    E2 = get_error(model, i2)
    r2 = E2 * y2

    # 重点：这一段的重点在于确定 alpha1, 也就是old a1，并送到take_step去analytically 优化
    # 下面条件之一满足，进入if开始找第二个alpha，送到take_step进行优化
    if ((r2 < -tol and alph2 < model.C) or (r2 > tol and alph2 > 0)):
        if len(model.alphas[(model.alphas != 0) & (model.alphas != model.C)]) > 1:
            # 选择Ei矩阵中差值最大的先进性优化
            # 要想|E1-E2|最大，只需要在E2为正时，选择最小的Ei作为E1
            # 在E2为负时选择最大的Ei作为E1
            if model.errors[i2] > 0:
                i1 = np.argmin(model.errors)
            elif model.errors[i2] <= 0:
                i1 = np.argmax(model.errors)

            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model

        # 循环所有非0 非C alphas值进行优化，随机选择起始点
        for i1 in np.roll(np.where((model.alphas != 0) & (model.alphas != model.C))[0],
                          np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model

        # a2确定的情况下，如何选择a1? 循环所有(m-1) alphas, 随机选择起始点
        for i1 in np.roll(np.arange(model.m), np.random.choice(np.arange(model.m))):
            # print("what is the first i1",i1)
            step_result, model = take_step(i1, i2, model)

            if step_result:
                return 1, model
    # 先看最上面的if语句，如果if条件不满足，说明KKT条件已满足，找其它样本进行优化，则执行下面这句，退出
    return 0, model


def fit(model):
    numChanged = 0
    examineAll = 1

    # loop num record
    # 计数器，记录优化时的循环次数
    loopnum = 0
    loopnum1 = 0
    loopnum2 = 0

    # 当numChanged = 0 and examineAll = 0时 循环退出
    # 实际是顺序地执行完所有的样本，也就是第一个if中的循环，
    # 并且 else中的循环没有可优化的alpha，目标函数收敛了： 在容差之内，并且满足KKT条件
    # 则循环退出，如果执行3000次循环仍未收敛，也退出
    # 重点：这段的重点在于确定 alpha2，也就是old a 2, 或者说alpha2的下标，old a2和old a1都是heuristically 选择
    while (numChanged > 0) or (examineAll):
        numChanged = 0
        if loopnum == 2000:
            break
        loopnum = loopnum + 1
        if examineAll:
            loopnum1 = loopnum1 + 1  # 记录顺序一个一个选择alpha2时的循环次数
            # # 从0,1,2,3,...,m顺序选择a2的，送给examine_example选择alpha1，总共m(m-1)种选法
            for i in range(model.alphas.shape[0]):
                examine_result, model = examine_example(i, model)
                numChanged += examine_result
        else:  # 上面if里m(m-1)执行完的后执行
            loopnum2 = loopnum2 + 1
            # loop over examples where alphas are not already at their limits
            for i in np.where((model.alphas != 0) & (model.alphas != model.C))[0]:
                examine_result, model = examine_example(i, model)
                numChanged += examine_result
        if examineAll == 1:
            examineAll = 0
        elif numChanged == 0:
            examineAll = 1
    print("loopnum012", loopnum, ":", loopnum1, ":", loopnum2)
    return model


# 从文档中读取数据，数据的格式为：
# X Y Z，其中Z为标签值，使用空格分割
def loadDataSet(file_path):
    data_arr, label_arr = [], []
    fr = open(file_path)
    for line in fr.readlines():
        arr = line.strip().split(' ')
        # 转化为浮点数，方便计算
        data_arr.append([float(arr[0]), float(arr[1])])
        label_arr.append(float(arr[2]))
    return data_arr, label_arr

# 生成测试数据，训练样本
X_train, y = loadDataSet('test.txt')
# StandardScaler()以及fit_transfrom函数的作用需要解释一下
scaler = StandardScaler()  # 数据预处理，使得经过处理的数据符合正态分布，即均值为0，标准差为1
# 训练样本异常大或异常小会影响样本的正确训练，如果数据的分布很分散也会影响
X_train_scaled = scaler.fit_transform(X_train, y)
y[y == 0] = -1

# set model parameters and initial values
C = 20.0
m = len(X_train_scaled)
initial_alphas = np.zeros(m)
initial_b = 0.0

# set tolerances
tol = 0.01  # error tolerance
eps = 0.01  # alpha tolerance

# Instaantiate model

model = SMOStruct(X_train_scaled, y, C, linear_kernel, initial_alphas, initial_b, np.zeros(m), user_linear_optim=True)
# print("model created ...")
# initialize error cache

initial_error = decision_function(model.alphas, model.y, model.kernel, model.X, model.X, model.b) - model.y
model.errors = initial_error
np.random.seed(0)


print("starting to fit...")
# 开始训练
output = fit(model)
# 绘制训练完，找到分割平面的图
fig, ax = plt.subplots()
plot_decision_boundary(output)

