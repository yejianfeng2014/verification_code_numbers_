# verification_code_numbers_
识别纯数字的验证码

项目介绍
    第一阶段 实现纯数字验证码的识别 准确率达到85%以上
    第二阶段 实现数字 大写字母 小写字母的验证码识别，准确率80%以上


目录结果介绍
    -model 存放纯数字识别的模型，由于模型较大，github 上传不上去，如果想使用训练好的模型，请去 百度网盘下载
    -number_alphabet 数字和字母的验证识别目录
        --
    -image_Create.py 生成随机验证码
    -pev.py 数据预处理的一些操作
    -trainCNN.py 核心代码，大部分功能在此实现
    -trianCNN_shell 在服务器上训练，屏蔽了部分图形的包


网络结构介绍：

    三层卷积网络，两层全连接网络
注意事项：
        1，loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
            激活函数选择上面的这个时，loss 在训练中不断变大所以改为了下面的
           loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=y))



