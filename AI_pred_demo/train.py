import numpy as np
from PIL import Image
import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import time
#from keras.utils import multi_gpu_model
from keras import optimizers

WIDTH = 80
HEIGHT = 80
FRAMES = 16

SEQUENCE = np.load('data.npz')['sequence_array']  # load array
print(SEQUENCE[0])
print('Data loaded.')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

NUMBER = len(SEQUENCE)

'''
i = 0
while i < NUMBER:
    if (i + 1) % 11 != 0:
        BASIC_SEQUENCE = np.append(BASIC_SEQUENCE, SEQUENCE[i])
        NEXT_SEQUENCE = np.append(NEXT_SEQUENCE, SEQUENCE[i+1])
    i += 1
    print(i)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
'''

# step =1
SEQUENCE = SEQUENCE.reshape(NUMBER, WIDTH, HEIGHT, 1)
# step =2
SEQUENCE_2 = []
for i in range(int(NUMBER / 2)):
    SEQUENCE_2.append(SEQUENCE[2 * i])

# step = 3
SEQUENCE_3 = []
for i in range(int(NUMBER / 3)):
    SEQUENCE_3.append(SEQUENCE[3 * i])

#def get_sequence()


SEQUENCE = SEQUENCE.reshape(NUMBER, WIDTH, HEIGHT, 1)
BASIC_SEQUENCE = np.zeros((NUMBER-FRAMES, FRAMES, WIDTH, HEIGHT, 1))
NEXT_SEQUENCE = np.zeros((NUMBER-FRAMES, FRAMES, WIDTH, HEIGHT, 1))


for i in range(FRAMES):
    print(i)
    BASIC_SEQUENCE[:, i, :, :, :] = SEQUENCE[i:i+NUMBER-FRAMES]
    NEXT_SEQUENCE[:, i, :, :, :] = SEQUENCE[i+1:i+NUMBER-FRAMES+1]



plt.imshow(BASIC_SEQUENCE[200][0].reshape(80, 80))
plt.show()
# build model


