import numpy as np

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']


def convert_to_gray(image):
    if len(image.shape) > 2:
        gray = np.mean(image, -1)  # 转成灰度图
        return gray
    else:
        return image


def find_index(x):
    alldata = number + alphabet + ALPHABET
    index = alldata.index(x)
    return index


def text2vec(text, max_text, char_size):
    vector = np.zeros(max_text * char_size)
    for i, c in enumerate(text):
        index = find_index(c)
        print(index)
        idx = i * char_size + index
        vector[idx] = 1
    return vector
