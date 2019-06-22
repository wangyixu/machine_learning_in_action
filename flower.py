import tensorflow as tf
import numpy as np
import os
from PIL import Image
import random

# 定义神经网络结构相关的参数
IMAGE_SIZE = 32
NUM_CHANNELS = 1

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512
NUM_LABELS = 5

OUTPUT_NODE = 5

# 定义模型训练相关的参数
BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 500
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "FLOWER_model/"
MODEL_NAME = "flower_model"


def load_dataset(filename):
    '''
    加载数据文件，存储为矩阵
    :param filename:
    :return:
    '''
    label_names = []
    images = []
    labels = []
    for dirpath, dirnames, filenames in os.walk('flower_photos'):
        for dir in dirnames:
            label_names.append(dir)
            print(dir)
            cnt = 0
            for pic in os.listdir(os.path.join(dirpath, dir)):
                image = io.imread('flower_photos/' + dir + '/' + pic)
                image = transform.resize(image, (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
                images.append(np.reshape(image, [IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS]))

                label = [0.0] * 5
                label[label_names.index(dir)] = 1
                labels.append(label)
                # if os.path.splitext(pic)[1] == '.jpg':
                cnt += 1

                '''
                img = Image.open(os.path.join(dirpath, dir) + '/' + pic)
                pix = img.load()
                # rint(img)
                reIm = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
                # im_arr = np.array(reIm.convert('L'))
                # print(im_arr.shape)
                # 模型的要求是黑底白字，但输入的图是白底黑字，所以需要对每个像素点的值改为255减去原值以得到互补的反色
                im_arr = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float)
                for i in range(IMAGE_SIZE):
                    for j in range(IMAGE_SIZE):
                        # print(pix[i, j])
                        # im_arr[i][j] = pix[i, j]  # r, g, b
                        r, g, b = pix[i, j]
                        gray = (r * 299 + g * 587 + b * 114 + 500) / 1000
                        im_arr[i][j] = gray * (1. / 255)

                img_raw = im_arr.reshape([IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS])
                # img_raw = tf.cast(img_raw, tf.float32) * (1. / 255)
                label = [0.0] * 5
                label[label_names.index(dir)] = 1

                # 将图片转换为需要的四个张量
                # nm_arr = im_arr.reshape([1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
                img_raw = img_raw.astype(np.float32)
                # print(img_ready)
                images.append(img_raw)
                labels.append(label)
                # print(img_raw)
                # print(label)
                print('------')
                '''
            print(cnt)
    print(label_names)
    print(len(images))
    print(len(images[0]))
    print(len(labels))
    print(len(labels[0]))
    return images, labels


def inference(input_tensor, train, regularizer):
    '''
    定义神经网络的前向传播过程
    :param input_tensor: 输入张量
    :param regularizer: 是否正则化
    :return:
    '''
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit


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
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference(x, True, regularizer)
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

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(TRAINING_STEPS):
            xs, ys = next_batch(train_data, train_labels, BATCH_SIZE)  # 100x256 100x10
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if step % 10 == 0:
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
        x = tf.placeholder(tf.float32, [
            len(test_data),
            IMAGE_SIZE,
            IMAGE_SIZE,
            NUM_CHANNELS], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE])
        y = inference(x, False, None)  # 前向传播得到预测结果 y

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
                reshaped_xs = np.reshape(test_data, (
                    len(test_data),
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    NUM_CHANNELS))
                accuracy_score = sess.run(accuracy, feed_dict={x: reshaped_xs, y_: test_labels})
                print("After %s training step(s), test accuracy = %g " % (global_step, accuracy_score))
            else:
                print('No checkpoint file found')

import glob
from skimage import io, transform

def read_image(path):
    #label_dir = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]#子文件目录
    label_dir = []
    for dirpath, dirnames, filenames in os.walk(path):
        label_dir = dirnames
        print(label_dir)
    images = []
    labels = []
    for index, folder in enumerate(label_dir):#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        for img in glob.glob(folder+'/*.png'):#获取指定目录下的所有图片
            print("reading the image:%s"%img)
            image = io.imread(img)
            image = transform.resize(image,(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
            print(image)
            print(index)
            images.append(image)
            labels.append(index)
    return np.asarray(images,dtype=np.float32),np.asarray(labels,dtype=np.int32)#array和asarray都可以将结构数据转化为ndarray，但是主要区别就是当数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，但asarray不会


if __name__ == '__main__':

    # 加载并随机打乱数据集
    images, labels = load_dataset("/flower_photos")

    # 使用相同的随机状态，对 images, labels 进行相同的 shuffle
    state = np.random.get_state()
    random.shuffle(images)
    np.random.set_state(state)
    random.shuffle(labels)
    # print(images[:10])
    # print(labels[:10])


    # 训练和测试
    train(images[:], labels[:])
    print()


    for _ in range(10):
        state = np.random.get_state()
        random.shuffle(images)
        np.random.set_state(state)
        random.shuffle(labels)
        test(images[:500], labels[:500])
    '''
    load_dataset('../flower_photos')
    '''