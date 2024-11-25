import Chamaeleo
from Chamaeleo.methods.fixed import *
import sys
import os
from math import *
import time
import re
from reedsolo import RSCodec

import subprocess

import random
import numpy as np
from PIL import Image
from collections import Counter
from scipy import signal
from scipy import ndimage
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

 

def mutate(string,rate):
    ins=0;dele=0;subi=0
    dna = list(string)
    random.seed(None)
    for index, char in enumerate(dna):
        if (random.random() <= rate):
            h=random.random()
            if(h<=0.8):
                subi+=1
                dna[index]=sub(dna[index])
            elif(h<=0.9):
                dele+=1
                dna[index]=""
            else:
                ins+=1
                dna[index]+=insert()
    return "".join(dna),subi,dele,ins



def sub(base):
    random.seed(None)
    suiji=random.randint(0,8)
    if(base=="A"):    
        if(suiji<3): return 'T'
        if(suiji<6): return 'C'
        if(suiji<9): return 'G'
    if(base=="G"):
        if(suiji<3): return 'T'
        if(suiji<6): return 'C'
        if(suiji<9): return 'A'
    if(base=="C"):
        if(suiji<3): return 'T'
        if(suiji<6): return 'G'
        if(suiji<9): return 'A'
    if(base=="T"):
        if(suiji<3): return 'G'
        if(suiji<6): return 'C'
        if(suiji<9): return 'A'

    
def insert():
    suiji=random.randint(0,3)
    if(suiji==1): return 'T'
    if(suiji==2): return 'C'
    if(suiji==3): return 'G'
    if(suiji==0): return 'A'

def sequence_error(sequence,rate):
    h=[];s=0;d=0;i=0
    for each in sequence:
        temp,subi,dele,ins=mutate(each, rate)
        s+=subi;d+=dele;i+=ins
        h.append(temp)
    return h

def split_string_by_length(string, length):
    result = []
    for i in range(0, len(string), length):
        result.append(string[i:i+length])
    return result



def encode(bit_segments):
    coding_scheme = Blawat()
    return coding_scheme.encode(bit_segments)
      

def image_to_bitstream(image_path):

    img = Image.open(image_path)
    img_arr = np.array(img)    
    bitstream = ''.join([f"{bin(pixel)[2:].zfill(8)}" for pixel in img_arr.flatten()])
    
    return bitstream
import os
import Chamaeleo
from Chamaeleo.methods.fixed import *
import numpy as np
from PIL import Image


def decode(sequence):
    coding_scheme = Blawat()
    return coding_scheme.decode(sequence)
      
def bitstream_to_image(bitstream, image_size):
    array_length = image_size[0] * image_size[1]
    
    image_array = np.array([int(bitstream[i:i+8], 2) for i in range(0, array_length*8, 8)], dtype=np.uint8)
    image_array = image_array.reshape(image_size[1], image_size[0])
    image = Image.fromarray(image_array)
    return image


def correct_length(group_sequence):
    correct_length_sequence=[]
    for l in group_sequence:
        length=len(l)
        if(length>150):
            while(length>150):
                l = l[:75] + l[76:]
                length-=1
        else:
            while(length<150):
                l = l[:75] +"C" +l[75:]
                length+=1
        correct_length_sequence.append(l)
    return correct_length_sequence

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as MSE
import os
import Chamaeleo
from Chamaeleo.methods.fixed import *
import numpy as np
from PIL import Image
import time    
if __name__ == "__main__":
    #h="fly_grey.jpg"
    #h="baboon_grey.jpg"
    h="lena.bmp"
    bitstream=image_to_bitstream(h)
    split_bitstream=split_string_by_length(bitstream,8)
    DNA=encode(split_bitstream)
    
    origin_sequence=""
    for i in DNA :
        origin_sequence+=(''.join(i))
    
    split_sequence_1=split_string_by_length(origin_sequence,150)
    
    
    
    test=0
    while(test<1):
        test+=1
    
    
        for error in [0.01,0.05]:

            muta_dna=sequence_error(split_sequence_1,error)
            
            
            correct_dna=correct_length(muta_dna)
            sequence=""
            for i in correct_dna:
                sequence+=i
            
            split_sequence=split_string_by_length(sequence,5)
            bitstream=decode(split_sequence)
            bit=""
            for i in bitstream:
                bit+=(''.join(i))
            image_size = (256, 256)  # 图像尺寸
            image = bitstream_to_image(bit[:524288], image_size)
            image.save("blawat_"+str(error)+".bmp")

    



