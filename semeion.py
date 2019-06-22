import tensorflow as tf
import numpy as np
import os

# 定义神经网络结构相关的参数
INPUT_NODE = 256
OUTPUT_NODE = 10
LAYER1_NODE = 500
# 定义模型训练相关的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="MNIST_model/"
MODEL_NAME="mnist_model"


def load_dataset(filename):
    '''
    加载数据文件，存储为矩阵
    :param filename:
    :return:
    '''
    fr = open(filename)
    data_matrix = []
    for line in fr.readlines():
        data_matrix.append(list(map(float, line.strip().split(" "))))
    return data_matrix


def get_weight_variable(shape, regularizer):
    '''通过tf.get_variable函数来获取变量。
    :param shape: 变量维度大小
    :param regularizer: 是否正则化
    :return: tf变量
    '''
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    '''
    定义神经网络的前向传播过程
    :param input_tensor: 输入张量
    :param regularizer: 是否正则化
    :return:
    '''
    with tf.variable_scope('layer1'):

        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2


def next_batch(train_data, train_target, batch_size):
    '''
    下一组 batch 大小的数据和标签
    :param train_data: 训练集数据
    :param train_target: 训练集标签
    :param batch_size:
    :return:
    '''
    index = [i for i in range(0, len(train_target))]
    np.random.shuffle(index)
    batch_data = []
    batch_target = []
    for i in range(0, batch_size):
        batch_data.append(train_data[index[i]])
        batch_target.append(train_target[index[i]])
    return batch_data, batch_target


def train(train_data, train_labels):
    '''
    定义训练过程。
    :param train_data: 训练集数据
    :param train_labels: 训练集标签
    :return:
    '''
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        len(train_data) / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = next_batch(train_data, train_labels, BATCH_SIZE)   # 100x256 100x10
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if step % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def test(test_data, test_labels):
    '''
    使用训练好的模型对测试集进行测试
    :param test_data: 测试集数据
    :param test_labels: 测试集标签
    :return:
    '''
    with tf.Graph().as_default() as g:
        # 给 x y_占位
        x = tf.placeholder(tf.float32, [None, INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE])
        y = inference(x, None)  # 前向传播得到预测结果 y

        # 实例化还原滑动平均的 saver
        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        # 计算准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            # 加载训练好的模型
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            # 如果已有 ckpt 模型则恢复会话、轮数、计算准确率
            if ckpt and ckpt.model_checkpoint_path:
                path = ckpt.model_checkpoint_path
                saver.restore(sess, path)
                global_step = path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict={x: test_data, y_: test_labels})
                print("After %s training step(s), test accuracy = %g " % (global_step, accuracy_score))
            else:
                print('No checkpoint file found')


if __name__ == '__main__':
    # 加载并随机打乱数据集
    data_matrix = load_dataset("semeion.data")
    np.random.shuffle(data_matrix)
    # 分割数据集矩阵为 images矩阵 和 labels矩阵
    images = [line[:256] for line in data_matrix]
    labels = [line[256:] for line in data_matrix]
    # 训练和测试
    train(images[200:], labels[200:])
    print()
    test(images[:200], labels[:200])




