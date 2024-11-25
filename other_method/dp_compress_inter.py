import cv2
import numpy as np
import sys
import os
import subprocess
from math import *
import time
import re
from collections import Counter
from PIL import Image
import random
from reedsolo import RSCodec

import re
import subprocess
import math
import numpy as np
from PIL import Image
import re
import numpy as np

import random
import random
import numpy as np
from PIL import Image
import re
from collections import Counter
import math
import os
from scipy import signal
from scipy import ndimage
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()
 

def mutate(string,rate):
    ins=0;dele=0;subi=0
    dna = list(string)
    random.seed(None)
    for index, char in enumerate(dna):
        if (random.random() <= rate):
            h=random.random()
            if(h<=0.33):
                subi+=1
                dna[index]=sub(dna[index])
            elif(h<=0.66):
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




def image_to_bitstream(image_path):


    img = Image.open(image_path)
    img_arr = np.array(img)
    
    
    
    bitstream = ''.join([f"{bin(pixel)[2:].zfill(8)}" for pixel in img_arr.flatten()])
    
    return bitstream


def bitstream_to_image(bitstream, image_size):
    array_length = image_size[0] * image_size[1]
    
    image_array = np.array([int(bitstream[i:i+8], 2) for i in range(0, array_length*8, 8)], dtype=np.uint8)
    image_array = image_array.reshape(image_size[1], image_size[0])
    image = Image.fromarray(image_array)
    return image

def split_string_by_length(string, length):
    result = []
    for i in range(0, len(string), length):
        result.append(string[i:i+length])
    return result
def dna_encode(binary):
    
    
    DNA_encoding = {"00": "A","01": "T","10": "C","11": "G"}
    binary_list = [binary[i: i+2] for i in range(0, len(binary), 2)]
    
    DNA_list = []
    for num in binary_list:
        for key in list(DNA_encoding.keys()):
            if num == key:
                DNA_list.append(DNA_encoding.get(key))

    return "".join(DNA_list)
def dna_decode(dna):
    DNA_encoding = {"A": "00","T": "01","C": "10","G": "11"}
    bin_list = []
    for num in dna:
        for key in list(DNA_encoding.keys()):
            if num == key:
                bin_list.append(DNA_encoding.get(key))

    return "".join(bin_list)




def image_to_bitstream(image_path):


    img = Image.open(image_path)
    img_arr = np.array(img)
    
    
    
    bitstream = ''.join([f"{bin(pixel)[2:].zfill(8)}" for pixel in img_arr.flatten()])
    
    return bitstream


def bitstream_to_image(bitstream, image_size):
    array_length = image_size[0] * image_size[1]
    
    image_array = np.array([int(bitstream[i:i+8], 2) for i in range(0, array_length*8, 8)], dtype=np.uint8)
    image_array = image_array.reshape(image_size[1], image_size[0])
    image = Image.fromarray(image_array)
    return image


def direct(a):
    mat = np.array(a)
    
    b = mat.transpose()
    
    return b

def correct_length(group_sequence):
    correct_length_sequence=[]
    for l in group_sequence:
        length=len(l)
        if(length>152):
            while(length>152):
                l = l[:76] + l[77:]
                length-=1
        else:
            while(length<152):
                l = l[:76] +"C" +l[76:]
                length+=1
        correct_length_sequence.append(l)
    return correct_length_sequence
from decimal import Decimal

def rs_encode_bitstream(binary_code): #add rs
    binary_code=binary_code.replace('A','00').replace('T','01').replace('C','10').replace('G','11')
   
    
    rsc = RSCodec(4)
    bytes_msg=bytes(int(binary_code[i:i+8],2)for i in range(0,len(binary_code),8))
    array_msg=bytearray(bytes_msg)
    array_msg=rsc.encode(array_msg)

    binary_code=''.join(format(x,'08b') for x in array_msg)
    
    return dna_encode(binary_code)
def image_to_bitstream(image_path):


    img = Image.open(image_path)
    img_arr = np.array(img)
    
    
    return img_arr



                
import numpy as np
import random
from PIL import Image
def dna_encode(binary):
    
    
    DNA_encoding = {"00": "A","01": "T","10": "C","11": "G"}
    binary_list = [binary[i: i+2] for i in range(0, len(binary), 2)]
    
    DNA_list = []
    for num in binary_list:
        for key in list(DNA_encoding.keys()):
            if num == key:
                DNA_list.append(DNA_encoding.get(key))

    return "".join(DNA_list)
def dna_decode(dna):
    DNA_encoding = {"A": "00","T": "01","C": "10","G": "11"}
    bin_list = []
    for num in dna:
        bin_list.append(DNA_encoding.get(num))

    return "".join(bin_list)
def split_string_by_length(string, length):
    result = []
    for i in range(0, len(string), length):
        result.append(string[i:i+length])
    return result

def rs_decode_bitstream(binary_code):
    
    binary_code=binary_code.replace('A','00').replace('T','01').replace('C','10').replace('G','11')
    
    
    rsc = RSCodec(4)
    #convert a string like ACTCA to an array of ints like [10, 2, 4]
    
    bytes_msg=bytes(int(binary_code[i:i+8],2)for i in range(0,len(binary_code),8))
    array_msg=bytearray(bytes_msg)
      
    data=rsc.decode(array_msg)
    binary_code=''.join(format(x,'08b') for x in data[0]) #转二进制
    
    return ''.join(str(int(binary_code[t:t+2],2)) for t in range(0, len(binary_code),2)).replace('0','A').replace('1','T').replace('2','C').replace('3','G')

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()
 
 
def ssim_1(img1, img2, cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))
 
def mssim(img1, img2):

    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2))/4.0
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = ssim_1(im1, im2, cs_map=True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter, 
                                                mode='reflect')
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter, 
                                                mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (np.prod(mcs[0:level-1]**weight[0:level-1])*
                    (mssim[level-1]**weight[level-1]))
def apply_median_filter(image):
    return cv2.medianBlur(image, 3)
from skimage.metrics import mean_squared_error as MSE
                
if __name__ == "__main__":
    #bitstream=image_to_bitstream("baboon_grey.jpg")
    #bitstream=image_to_bitstream("fly_grey.jpg")
    bitstream=image_to_bitstream("lena.bmp")
    out=open("matrix.txt","w")

    for i in range(len(bitstream)):
        if(i%2==0):
            for j in range(len(bitstream[0])):
                out.write(str(bitstream[i][j])+"\t")
        else:
            for j in range(len(bitstream[0])-1,-1,-1):
                out.write(str(bitstream[i][j])+"\t")
    out.close()

    
    subprocess.run(["g++", "dp-compress.cpp", "-o", "test"], capture_output=True, text=True)
    result=subprocess.run(["./test"],capture_output=True, text=True)
    print(result)
    
    with open("matrix.txt", 'r') as file:   #得到图像弓形数组向量
        for line in file:
            line = line.strip()
            data = line.split("\t")

    count=0
    bits_01="";bits_signal=""
    with open("signal_pixel.txt", 'r') as file:  #covert matrix to bitstream 
        for line in file:
            line = line.strip()
            tmp = line.split(" ")
            number_of_bits=bin(int(tmp[1])-1)[2:].zfill(3)  #number_of_bits
            count_temp=bin(int(tmp[0])-1)[2:].zfill(8)      #count
            #4-bits(number_of_bits)+8-bits(count)+dynamic-bits(pixel)
            bits_signal=bits_signal+number_of_bits+count_temp
            
            for i in range(int(tmp[0])):
                binary_str = bin(int(data[count+i]))[2:]
                binary_str = '0' * (int(tmp[1]) - len(binary_str)) + binary_str  
                
                bits_01+=binary_str    
            count+=int(tmp[0])
    base=["A","C","G","T"]
    data_dna=dna_encode(bits_01)
    while(len(data_dna)%152!=0):
        
        data_dna += random.choice(base)
        
    signal_dna=dna_encode(bits_signal)
    
    while(len(signal_dna)%56!=0):
        signal_dna+=random.choice(base)
    
    numbers = list(range(0, len(data_dna)))
    random.seed(10)
    random.shuffle(numbers)
    new=""
    for i in range(len(numbers)):
        new+=data_dna[numbers[i]]#
    file_name="lena.txt"

    stl = split_string_by_length(signal_dna, 56)

        
    stl = split_string_by_length(new, 152)
    test=0

    while(test<10000):
        print(test)
        test+=1
        for error in [0.01,0.02,0.03,0.04,0.05]:
            out=open("dpid_"+str(error)+".txt","a")
            muta_dna=sequence_error(stl,error)
            
            
            correct_dna=correct_length(muta_dna)
            
            
           
            signal_dna_2="";data_dna_2=""

            for line in correct_dna:
                
                data_dna_2+=line

            
            
            numbers = list(range(0, len(data_dna_2)))
            random.seed(10)
            random.shuffle(numbers)
            dna_new=list(range(0, len(data_dna_2)))
            for i in range(len(numbers)):

                dna_new[numbers[i]]=data_dna_2[i]
            data_dna_3="".join(dna_new)#
            cnt=0
            for i in range(len(data_dna_3)):
                if(new[i]==data_dna_3[i]):
                    cnt+=1
            bits_02=dna_decode(data_dna_3)
            
            
            
            d=[];count=0  ;bits_signal_count=0   
            
            while(count<len(bits_01)):
                
                number_of_bits=int(bits_signal[bits_signal_count:bits_signal_count+3],2)+1;count_temp=int(bits_signal[bits_signal_count+3:bits_signal_count+11],2)+1
                #print(number_of_bits,count_temp)
                for i in range(count_temp):
                    d.append((int(bits_02[count+i*number_of_bits:count+(i+1)*number_of_bits],2)))
                count+=(number_of_bits*count_temp)
                
                bits_signal_count+=11
                
            col=256;row=256;kk=0
            matrix = np.empty(shape=(row, col),dtype=int)
            for i in range(col):
                if(i%2==0):
                    for j in range(row):
                        matrix[i][j]=d[kk] 
                        kk+=1
                else:
                    for j in range(row-1,-1,-1):
                        matrix[i][j]=d[kk] 
                        kk+=1
                 
            image = Image.fromarray(np.uint8(matrix))
            image.save('dp_inter_'+str(error*100)+'%.bmp')
            
            h='dp_inter_'+str(error*100)+"_median.bmp"
            image=cv2.imread('dp_inter_'+str(error*100)+'%.bmp')
            median = apply_median_filter(image)
            cv2.imwrite(h, median)
            
            img1 = cv2.imread("lena.bmp",cv2.IMREAD_GRAYSCALE)
            img3 = cv2.imread(h,cv2.IMREAD_GRAYSCALE)
            out.write(str(round(MSE(img1, img3),3))+"\t")
            out.write(str(round(psnr(img1, img3),3))+"\t")
            out.write(str(round(ssim(img1, img3),3))+"\t")
                                        
            out.write(str(round(mssim(img1, img3),3))+"\t")

            out.write("\n")
            out.close()





