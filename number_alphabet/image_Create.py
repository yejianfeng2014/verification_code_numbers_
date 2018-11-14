'''

 数字 大写字母 ，小写字母 10 +26+26 =62
'''

import random
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np

number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']


# 获得随机四个数字
def random_alphabet(char_set=number+alphabet + ALPHABET, char_size=4):
    chat_test = []

    for i in range(char_size):
        chat = random.choice(char_set)
        chat_test.append(chat)

        # print(chat_test)
    return chat_test


# 生成字符串 和验证码的图片
def get_text_image():
    image = ImageCaptcha()
    text = random_alphabet()  # 调用生成的四个数字
    for i in range(len(text)):
        text[i] = str(text[i])
    new_text = ''.join(text)
    text = new_text
    captcha = image.generate(text)

    captcha_image = PIL.Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return text, captcha_image


def show():
    text, image = get_text_image()

    # imag
    f = plt.figure()
    ax = f.add_subplot(111)
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    show()
