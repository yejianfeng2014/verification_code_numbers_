import numpy as np
import tensorflow as tf

from number_alphabet import image_Create, pev

image_height = 60
image_width = 160
keep_cell = 0.8
Max_text = 248  # 四个 乘以 62
Char_size = 4

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']


# 定义网络结果
def train_cell(x, w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(x, shape=[-1, image_height, image_width, 1])
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))

    # 卷积
    conv1 = tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME')
    # 偏置
    add_1 = tf.nn.bias_add(conv1, b_c1)
    # 激活函数
    relu_1 = tf.nn.relu(add_1)
    # 池化 大小 2*2 步长 2
    pool_1 = tf.nn.max_pool(relu_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    dropout_1 = tf.nn.dropout(pool_1, keep_cell)

    # 第二层 输入32 输出64
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.conv2d(dropout_1, w_c2, strides=[1, 1, 1, 1], padding='SAME')
    add_2 = tf.nn.bias_add(conv2, b_c2)
    relu_2 = tf.nn.relu(add_2)
    pool_2 = tf.nn.max_pool(relu_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    dropout_2 = tf.nn.dropout(pool_2, keep_cell)

    # 第三层  输入 64 输出64
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(dropout_2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_cell)

    # 全连接层 这个是个关键 输入参数需要计算 ，输出1024这个可以随意改变
    # 8 的计算过程 卷积层不改变大小，持有池化层改变大小，所以按照池化层计算 原始图形60 *160
    # 经过一次 60/2 = 30  第二次 30/2 = 15 第三次 15不够所以补充1为16  16/2 =8
    # 20的计算过程
    # 经过一次 160/2 = 80  第二次  80/2 = 40   40/2 =20
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))

    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])

    relu = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))

    dropout = tf.nn.dropout(relu, keep_cell)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, Max_text * Char_size]))

    b_out = tf.Variable(b_alpha * tf.random_normal([Max_text * Char_size]))

    out = tf.add(tf.matmul(dropout, w_out), b_out)

    return out


def train_CNN(x, y):


    output = train_cell(x)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=y))
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))

    optimer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, Max_text , Char_size])

    max_idx_p = tf.argmax(predict, 2)

    max_idx_l = tf.argmax(tf.reshape(y, [-1, Max_text, Char_size]), 2)

    correct_predict = tf.equal(max_idx_l, max_idx_p)

    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        _check_restore_parameters(sess,saver) # 从历史中恢复继续训练

        step = 1
        while True:
            batch_x, batch_y = get_next_batch(64)

            _, train_loss = sess.run([optimer, loss], feed_dict={x: batch_x, y: batch_y, keep_cell: 0.75})

            print('step', step, 'loss', train_loss)

            # 每一百步计算一次准确率

            if step % 10 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)

                acc = sess.run(accuracy, feed_dict={x: batch_x_test, y: batch_y_test, keep_cell: 1})

                print("step number", step, "step", "acc:", acc)

                if acc > 0.86:
                    saver.save(sess, "./model/crack_capcha.model", global_step=step)
                    break

            # if step > 1000:
            #     saver.save(sess, "./model/crack_capcha.model", global_step=step)
            #
            #     break

            step = step + 1


# 检查是否已经训练过
def _check_restore_parameters(sess, saver):
    """ 如果以前训练过，直接从默认的路径吓恢复，可以节约训练时间. """
    ckpt = tf.train.get_checkpoint_state('./model/checkpoint')
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the Chatbot")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the Chatbot")



def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, image_height * image_width])

    batch_y = np.zeros([batch_size, Max_text * Char_size])

    for i in range(batch_size):
        text, image = image_Create.get_text_image()

        image = pev.convert_to_gray(image)

        batch_x[i, :] = image.flatten() / 255

        batch_y[i, :] = pev.text2vec(text, Max_text, Char_size)

    return batch_x, batch_y
    pass


def crack_captcha(image):
    save_path = './../model/checkpoint'
    out_put = train_cell(x)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        predict = tf.argmax(tf.reshape(out_put, [-1, Max_text, Char_size]), 2)
        text_list = sess.run(predict, feed_dict={x: [image], keep_cell: 1})
        text = text_list[0].tolist()
        return text


if __name__ == '__main__':
    print('Welcome to recognition v0.1 !')
    print('TensorFlow detected: v{}'.format(tf.__version__))
    flag = 0
    if flag == 0:

        text, image = image_Create.get_text_image()
        image_height, image_width = (60, 160)
        char_set = number + alphabet + ALPHABET
        Max_text = len(text)
        Char_size = len(char_set)

        x = tf.placeholder(tf.float32, [None, image_height * image_width])

        y = tf.placeholder(tf.float32, [None, Max_text * Char_size])

        keep_cell = tf.placeholder(tf.float32)

        train_CNN(x, y)

    # if flag == 1:  # 预测
    #     image_height = 60
    #     image_width = 160
    #     char_set = number + alphabet + ALPHABET
    #     Char_size = len(char_set)
    #
    #     text, image = image_Create.get_text_image()
    #
    #     f = plt.figure()
    #     ax = f.add_subplot(111)
    #     ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    #     plt.imshow(image)
    #
    #     plt.show()
    #
    #     Max_text = len(text)
    #     image = pev.convert_to_gray(image)
    #     image = image.flatten() / 255
    #
    #     x = tf.placeholder(tf.float32, [None, image_height * image_width])
    #     y = tf.placeholder(tf.float32, [None, Max_text * Char_size])
    #     keep_cell = tf.placeholder(tf.float32)  # dropout
    #
    #     predict_text = crack_captcha(image)
    #     print("正确: {}  预测: {}".format(text, predict_text))
