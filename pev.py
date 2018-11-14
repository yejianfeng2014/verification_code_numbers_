import numpy as np


def convert_to_gray(image):
    if len(image.shape) > 2:
        gray = np.mean(image, -1)  # 转成灰度图
        return gray
    else:
        return image


def text2vec(text, max_text, char_size):
    text_len = len(text)
    vector = np.zeros(max_text * char_size)
    for i, c in enumerate(text):
        idx = i * char_size + int(c)
        vector[idx] = 1
    return vector
