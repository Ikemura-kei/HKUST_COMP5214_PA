from array import array
import struct
import numpy as np
import cv2 as cv

def read_mnist_data(img_path, label_path):
    # read images
    with open(img_path, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())    

    imgs = []
    
    for i in range(size):
        imgs.append([0] * rows * cols)
        
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(28, 28)
        imgs[i][:] = img 

    imgs = np.array(imgs) 
    
    # read labels
    labels = []
    with open(label_path, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())   
        
    return imgs, labels