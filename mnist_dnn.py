import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random

num_pic = 1
batch_size = 32
epochs = 10

zero_list = os.listdir("./mnist/0")
one_list = os.listdir("./mnist/1")
two_list = os.listdir("./mnist/2")
three_list = os.listdir("./mnist/3")
four_list = os.listdir("./mnist/4")
five_list = os.listdir("./mnist/5")
six_list = os.listdir("./mnist/6")
seven_list = os.listdir("./mnist/7")
eight_list = os.listdir("./mnist/8")
nine_list = os.listdir("./mnist/9")

number = [zero_list, one_list, two_list, three_list, four_list,
                   five_list, six_list, seven_list, eight_list, nine_list]

for num in number:
    random.shuffle(num)

# randomにファイルを1~10枚ずつのリストを作成
def set_num_list(num_list, num_pic):
    index = 0
    tmp1 = 0
    tmp2 = 0
    num_file_list = []
    while (index<len(num_list)):
        tmp2 += tmp1
        if(index+num_pic>=len(num_list)):
            tmp1 = len(num_list) - index
            index += tmp1
        else:
            tmp1 = int(round(np.random.random()*(num_pic-1))+1)
            index += tmp1
            num_file_list.append(num_list[tmp2:tmp1+tmp2])
    return num_file_list

def set_num_array(num_file_list, num_str, num_pic):
    num_array = []
    for num_file in num_file_list:
        tmp = []
        if len(num_file) < num_pic:
            for num in num_file:
                tmp.append(np.asarray(Image.open("./mnist/" + num_str + "/" + num)).reshape(28*28).tolist())
            e = num_pic - len(num_file)
            for i in range(e):
                tmp.append(np.zeros([28*28]).tolist())
        else:
            for num in num_file:
                tmp.append(np.asarray(Image.open("./mnist/" + num_str + "/" + num)).reshape(28*28).tolist())
        num_array.append(np.array(tmp).reshape(num_pic*28*28).tolist())
    num_array = np.array(num_array)
    return num_array

zero_file_list = set_num_list(zero_list, num_pic)
one_file_list = set_num_list(one_list, num_pic)
two_file_list = set_num_list(two_list, num_pic)
three_file_list = set_num_list(three_list, num_pic)
four_file_list = set_num_list(four_list, num_pic)
five_file_list = set_num_list(five_list, num_pic)
six_file_list = set_num_list(six_list, num_pic)
seven_file_list = set_num_list(seven_list, num_pic)
eight_file_list = set_num_list(eight_list, num_pic)
nine_file_list = set_num_list(nine_list, num_pic)

zero_array = set_num_array(zero_file_list, "0", num_pic)
one_array = set_num_array(one_file_list, "1", num_pic)
two_array = set_num_array(two_file_list, "2", num_pic)
three_array = set_num_array(three_file_list, "3", num_pic)
four_array = set_num_array(four_file_list, "4", num_pic)
five_array = set_num_array(five_file_list, "5", num_pic)
six_array = set_num_array(six_file_list, "6", num_pic)
seven_array = set_num_array(seven_file_list, "7", num_pic)
eight_array = set_num_array(eight_file_list, "8", num_pic)
nine_array = set_num_array(nine_file_list, "9", num_pic)

num_array = [zero_array, one_array, two_array, three_array, four_array,
                        five_array, six_array, seven_array, eight_array, nine_array]

tmp = [i for i in range(10)]
tmp_array = keras.utils.to_categorical(tmp)

X = np.concatenate((zero_array, one_array, two_array, three_array, four_array,
                                     five_array, six_array, seven_array, eight_array, nine_array)) / 255

y = []
for i, n_a in enumerate(num_array):
    for j in range(n_a.shape[0]):
        y.append(tmp_array[i])
        
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

n_in = X_train.shape[-1]
n_out = y_train.shape[-1]

model = Sequential()
model.add(Dense(512, input_dim=num_pic*28*28))
model.add(Activation('sigmoid'))
model.add(Dense(512))
model.add(Activation('sigmoid'))
model.add(Dense(512))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,)

test = model.evaluate(X_test, y_test, batch_size=32)

print("\n\nTest Acc: %f"test[1])
