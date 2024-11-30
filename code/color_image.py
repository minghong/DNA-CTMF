
import time
import numpy as np
import random
from collections import Counter
import numpy as np
from PIL import Image
import copy

import signal as si
import numpy as np
import random
from collections import Counter
import numpy as np
from PIL import Image


from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as MSE
import cv2

from PIL import Image
import numpy as np
from scipy import signal
from scipy import ndimage

 

class TimeoutException(BaseException):
    pass
 
def bit_to_image(img, red_image , green_image ,blue_image):
    img = Image.open(img)
    red_channel, green_channel, blue_channel = img.split()

    wigth, height = img.size
    red_image = np.reshape(red_image, np.array(red_channel).shape)
    green_image = np.reshape(green_image, np.array(green_channel).shape)
    blue_image = np.reshape(blue_image, np.array(blue_channel).shape)

    red = Image.fromarray(red_image.astype(np.uint8))
    green = Image.fromarray(green_image.astype(np.uint8))
    blue = Image.fromarray(blue_image.astype(np.uint8))

    reconstructed_image = Image.merge("RGB", (red, green, blue))

    return reconstructed_image

def set_timeout(seconds):

    signal.signal(signal.SIGALRM, timeout_handler)

    signal.alarm(seconds)
 
 

def timeout_handler(signum, frame):
    raise TimeoutException("exceed_time")


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

def apply_mean_filter(image):
    return cv2.blur(image, (3, 3))

def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (3, 3), 1.0)

def apply_median_filter(image):
    return cv2.medianBlur(image, 3)

class TimeoutException(BaseException):
    pass
 
 

def set_timeout(seconds):

    si.signal(si.SIGALRM, timeout_handler)

    si.alarm(seconds)



def dfs(i):  
    if(i==len(step)):
        if(sum(matrix_correct_tmp)+len(sequence)==150):           
            matrix_correct.append(copy.deepcopy(matrix_correct_tmp))            
        return
    else:
        if(len(step[i])==1):
            matrix_correct_tmp[i]=step[i][0];dfs(i+1)
        elif(len(step[i])==2):
            matrix_correct_tmp[i]=step[i][0];dfs(i+1)
            matrix_correct_tmp[i]=step[i][1];dfs(i+1)

    return

def mutate(string,rate):

    dna = list(string)
    random.seed(None)
    for index, char in enumerate(dna):
        if (random.random() <= rate):
            h=random.random()
            if(h<=1/3):

                dna[index]=sub(dna[index])
            elif(h<=2/3):

                dna[index]=""
            else:

                dna[index]+=insert()
    return "".join(dna)



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
        temp=mutate(each, rate)

        h.append(temp)
    return h

def hm_distance(str1, str2):
    count = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            count += 1
    return count
def merge_intervals_by_divide_conquer(intervals):
    if not intervals:
        return []
 

    intervals.sort(key=lambda x: x[0])

    def merge_intervals(intervals):
        if len(intervals) <= 1:
            return intervals
 
        mid = len(intervals) // 2
        left_intervals = merge_intervals(intervals[:mid])
        right_intervals = merge_intervals(intervals[mid:])
        return merge_sorted_intervals(left_intervals, right_intervals)

    def merge_sorted_intervals(left, right):
        merged = []
        i, j = 0, 0
        while i < len(left) and j < len(right):
            if left[i][1] < right[j][0]:
                merged.append(left[i])
                i += 1
            elif right[j][1] < left[i][0]:
                merged.append(right[j])
                j += 1
            else:
                start = min(left[i][0], right[j][0])
                end = max(left[i][1], right[j][1])
                merged.append([start, end])
                i += 1
                j += 1
 

        while i < len(left):
            merged.append(left[i])
            i += 1
        while j < len(right):
            merged.append(right[j])
            j += 1
 
        return merged
 

    return merge_intervals(intervals)


def gc(sequence):
    dic=dict(Counter(sequence))
    for k in ["A","T","G","C"]:
        dic.setdefault(k,0)
    return (dic["G"]+dic["C"])/(dic["A"]+dic["T"]+dic["G"]+dic["C"])
def test(string):
    if("AAA" in string):
        return False;
    if("TTT" in string):
        return False;
    if("CCC" in string):
        return False;
    if("GGG" in string):
        return False;
    return True

def image_to_bitstream(image_path):
    img = Image.open(image_path)
    red_channel, green_channel, blue_channel = img.split()

    img_np = np.array(red_channel)
    img_np1 = np.array(green_channel)
    img_np2 = np.array(blue_channel)
    red = img_np.flatten()
    green = img_np1.flatten()
    blue = img_np2.flatten()

    return red,green,blue

def split_string_by_length(string, length):
    result = []
    for i in range(0, len(string), length):
        result.append(string[i:i+length])
    return result 


def subtract_intervals(big_interval, small_intervals):
    big_list = list(range(big_interval[0], big_interval[1]+1))
    result = [i for i in big_list if all(i < s or i > e for s, e in small_intervals)]
    return result
def merge_sequence(wrong,right,right_num,wrong_num):
    correct=""
    right_len=len(right)
    wrong_len=len(wrong)
    i=0;j=0
    if(right_num>wrong_num):
        correct+=wrong[j];j+=1
    while(i < right_len and j < wrong_len):
        correct=correct+right[i]+wrong[j]
        i+=1;j+=1
    if(i==right_len and j < wrong_len):
        correct+=wrong[j]
    if(i<right_len and j == wrong_len):
        correct+=right[i]
    return correct
def error_combin_merge(ints):
    
    dd = {}
    a = [min(ints)]
    flag=0

    tt = []
    if(len(ints)==1):
        tt.append(ints)
        return tt
    for i in range(len(ints)-1):
        if ints[i+1] == ints[i] + 1 :
            a.append(ints[i+1])
            flag=1
        else:
            if(flag==0):
                tt.append(a)
                flag=1
            a = [ints[i+1]]
        tt.append(a)
    for x in tt:
        if not x[0] in dd:
            dd[x[0]] = x
        else:dd[x[0]] = sorted(list(set(dd[x[0]] + x)))
    return list(dd.values())


def correct_sub(wrong):
    xx=""
    for i in range(0,len(wrong),coding_table_len):
        if(wrong[i:i+coding_table_len] in map_rule):
            xx+=wrong[i:i+coding_table_len]
        else:

            minn=10000;tmp_minn=10000;
            for temp in coding_median:
                minn=min(hm_distance(wrong[i:i+coding_table_len],temp),minn)
                if(minn<tmp_minn):
                    tmp_minn=minn;tmp_xx=temp
            xx+=tmp_xx
    return xx
            
    
def correct_dele(wrong,num):
    temp_wrong=wrong
    
    tmp_corect="";tmp_haiming=10000
    if(num==1):
        for pos_1 in range(len(temp_wrong)):
            wrong=temp_wrong[:pos_1]+temp_wrong[pos_1+1:]
            haiming_1=0;flag=1
            for i in range(0,len(wrong),coding_table_len):
                xxxx=wrong[i:i+coding_table_len]
                if(gc(xxxx)<0.4 or gc(xxxx)>0.6):
                    flag=0; break
                if(xxxx in map_rule):
                    haiming_1+=0
                else:  
                    minn=10000
                    for temp in coding_median:
                        minn=min(hm_distance(xxxx,temp),minn)
                    haiming_1+=minn
            if(haiming_1==0 and flag==1):
                return wrong
            if(haiming_1<tmp_haiming and flag==1):
                tmp_haiming=haiming_1;tmp_corect=wrong
    if(num==2):
        for pos_1 in range(len(temp_wrong)):
            t_wrong=temp_wrong[:pos_1]+temp_wrong[pos_1+1:]
            for pos_2 in range(pos_1,len(t_wrong)):
                wrong=t_wrong[:pos_2]+t_wrong[pos_2+1:]
                haiming_1=0;flag=1
                for i in range(0,len(wrong),coding_table_len):
                    xxxx=wrong[i:i+coding_table_len]
                    if(gc(xxxx)<0.4 or gc(xxxx)>0.6):
                        flag=0; break
                    if(xxxx in map_rule):
                        haiming_1+=0
                    else:  
                        minn=10000
                        for temp in coding_median:
                            minn=min(hm_distance(xxxx,temp),minn)
                        haiming_1+=minn
                if(haiming_1==0 and flag==1):
                    return wrong
                if(haiming_1<tmp_haiming and flag==1):
                    tmp_haiming=haiming_1;tmp_corect=wrong
    if(num==3):
        for pos_1 in range(len(temp_wrong)):
            t_wrong=temp_wrong[:pos_1]+temp_wrong[pos_1+1:]
            for pos_2 in range(pos_1,len(t_wrong)):
                tt_wrong=t_wrong[:pos_2]+t_wrong[pos_2+1:]
                for pos_3 in range(pos_2,len(tt_wrong)):
                    wrong=tt_wrong[:pos_3]+tt_wrong[pos_3+1:]
                    if(test(wrong)==False):
                        continue                    
                    haiming_1=0;flag=1
                    for i in range(0,len(wrong),coding_table_len):
                        xxxx=wrong[i:i+coding_table_len]
                        if(gc(xxxx)<0.4 or gc(xxxx)>0.6):
                            flag=0; break
                        if(xxxx in map_rule):
                            haiming_1+=0
                        else:  
                            minn=10000
                            for temp in coding_median:
                                minn=min(hm_distance(xxxx,temp),minn)
                            haiming_1+=minn
                    if(haiming_1==0 and flag==1):
                        return wrong
                    if(haiming_1<tmp_haiming and flag==1):
                        tmp_haiming=haiming_1;tmp_corect=wrong
    if(num==4):
        for pos_1 in range(len(temp_wrong)):
            t_wrong=temp_wrong[:pos_1]+temp_wrong[pos_1+1:]
            for pos_2 in range(pos_1,len(t_wrong)):
                tt_wrong=t_wrong[:pos_2]+t_wrong[pos_2+1:]
                for pos_3 in range(pos_2,len(tt_wrong)):
                    ttt_wrong=tt_wrong[:pos_3]+tt_wrong[pos_3+1:]
                    for pos_4 in range(pos_3,len(ttt_wrong)):
                        wrong=ttt_wrong[:pos_4]+ttt_wrong[pos_4+1:]
                        
                        if(test(wrong)==False):
                            continue                        
                        haiming_1=0;flag=1
                        for i in range(0,len(wrong),coding_table_len):
                            xxxx=wrong[i:i+coding_table_len]
                            if(gc(xxxx)<0.4 or gc(xxxx)>0.6):
                                flag=0; break
                            if(xxxx in map_rule):
                                haiming_1+=0
                            else:  
                                minn=10000
                                for temp in coding_median:
                                    minn=min(hm_distance(xxxx,temp),minn)
                                haiming_1+=minn
                        if(haiming_1==0 and flag==1):
                            return wrong
                        if(haiming_1<tmp_haiming and flag==1):
                            tmp_haiming=haiming_1;tmp_corect=wrong
    if(tmp_corect==""):
        return (dic[int(median)]*(int((len(temp_wrong)-num)/5)))
    return correct_sub(tmp_corect)


def correct_ins(wrong,num):
    
    temp_wrong=wrong
    
    tmp_corect="";tmp_haiming=10000
    if(num==1):
        for pos_1 in range(len(temp_wrong)):
            for base_1 in base:
                wrong=temp_wrong[:pos_1]+base_1+temp_wrong[pos_1:]
                if(test(wrong)==False):
                    continue                
                haiming_1=0;flag=1
                for i in range(0,len(wrong),coding_table_len):
                    xxxx=wrong[i:i+coding_table_len]
                    if(gc(xxxx)<0.4 or gc(xxxx)>0.6):
                        flag=0; break
                    if(xxxx in map_rule):
                        haiming_1+=0
                    else:  
                        minn=10000
                        for temp in coding_median:
                            minn=min(hm_distance(xxxx,temp),minn)
                        haiming_1+=minn
                if(haiming_1==0 and flag==1):
                    return wrong
                if(haiming_1<tmp_haiming and flag==1):
                    tmp_haiming=haiming_1;tmp_corect=wrong
    if(num==2):
        for pos_1 in range(len(temp_wrong)):
            for base_1 in base:
                t_wrong=temp_wrong[:pos_1]+base_1+temp_wrong[pos_1:]
                for pos_2 in range(pos_1,len(wrong)):
                    for base_2 in base:
                        wrong=t_wrong[:pos_2]+base_2+t_wrong[pos_2:]
                        if(test(wrong)==False):
                            continue                       
                        haiming_1=0;flag=1
                        for i in range(0,len(wrong),coding_table_len):
                            xxxx=wrong[i:i+coding_table_len]
                            if(gc(xxxx)<0.4 or gc(xxxx)>0.6):
                                flag=0; break
                            if(xxxx in map_rule):
                                haiming_1+=0
                            else:  
                                minn=10000
                                for temp in coding_median:
                                    minn=min(hm_distance(xxxx,temp),minn)
                                haiming_1+=minn
                        if(haiming_1==0 and flag==1):
                            return wrong
                        if(haiming_1<tmp_haiming and flag==1):
                            tmp_haiming=haiming_1;tmp_corect=wrong
    if(num==3):
        for pos_1 in range(len(temp_wrong)):
            for base_1 in base:
                t_wrong=temp_wrong[:pos_1]+base_1+temp_wrong[pos_1:]
                for pos_2 in range(pos_1,len(t_wrong)):
                    for base_2 in base:
                        tt_wrong=t_wrong[:pos_2]+base_2+t_wrong[pos_2:]
                        for pos_3 in range(pos_2,len(tt_wrong)):
                            for base_3 in base:
                                wrong=tt_wrong[:pos_3]+base_3+tt_wrong[pos_3:]
                                if(test(wrong)==False):
                                    continue                                
                                haiming_1=0;flag=1
                                for i in range(0,len(wrong),coding_table_len):
                                    xxxx=wrong[i:i+coding_table_len]
                                    if(gc(xxxx)<0.4 or gc(xxxx)>0.6):
                                        flag=0; break
                                    if(xxxx in map_rule):
                                        haiming_1+=0
                                    else:  
                                        minn=10000
                                        for temp in coding_median:
                                            minn=min(hm_distance(xxxx,temp),minn)
                                        haiming_1+=minn
                                if(haiming_1==0 and flag==1):
                                    return wrong
                                if(haiming_1<tmp_haiming and flag==1):
                                    tmp_haiming=haiming_1;tmp_corect=wrong
    if(num==4):
        for pos_1 in range(len(temp_wrong)):
            for base_1 in base:
                t_wrong=temp_wrong[:pos_1]+base_1+temp_wrong[pos_1:]
                for pos_2 in range(pos_1,len(t_wrong)):
                    for base_2 in base:
                        tt_wrong=t_wrong[:pos_2]+base_2+t_wrong[pos_2:]
                        for pos_3 in range(pos_2,len(tt_wrong)):
                            for base_3 in base:
                                ttt_wrong=tt_wrong[:pos_3]+base_3+tt_wrong[pos_3:]
                                for pos_4 in range(pos_3,len(ttt_wrong)):
                                    for base_4 in base:
                                        wrong=ttt_wrong[:pos_4]+base_4+ttt_wrong[pos_4:]
                                        if(test(wrong)==False):
                                            continue
                                        haiming_1=0;flag=1
                                        for i in range(0,len(wrong),coding_table_len):
                                            xxxx=wrong[i:i+coding_table_len]
                                            if(gc(xxxx)<0.4 or gc(xxxx)>0.6):
                                                flag=0; break
                                            if(xxxx in map_rule):
                                                haiming_1+=0
                                            else:  
                                                minn=10000
                                                for temp in coding_median:
                                                    minn=min(hm_distance(xxxx,temp),minn)
                                                haiming_1+=minn
                                        if(haiming_1==0 and flag==1):
                                            return wrong
                                        if(haiming_1<tmp_haiming and flag==1):
                                            tmp_haiming=haiming_1;tmp_corect=wrong
    if(tmp_corect==""):
        return (dic[int(median)]*(int((len(temp_wrong)+num)/5)))
    return correct_sub(tmp_corect)
def handle_timeout(signum, frame):
    raise Exception("Timeout!")
    
coding_table_len=5
base=["A","C","G","T"]
dic={}
file=open("coding_table.txt")
i=0;map_rule=[]
for line in file.readlines():
   dic[i]=line.rstrip("\n")
   
   map_rule.append(dic[i])
   
   i+=1

image_name="Lisa.bmp"
red,green,blue=image_to_bitstream(image_name)
swapped_dict = {value: key for key, value in dic.items()}
  


for error in [0.01,0.02,0.03,0.04,0.05]:
    correct_pixel_x=[]
    for pixel in [red,green,blue]:
        string=""

        for i in pixel:
            string+=dic[i]

        each=split_string_by_length(string, 150)
        #模拟DNA存储
        #out=open(str(error)+"_index.txt","a")
        start=time.time()
        iiiii=0;new_sequence="";
        while(iiiii<(len(each)-1)):
            try:
                sequence="".join(sequence_error(each[iiiii], error));iiiii+=1
                    #找到所有5碱基的起点，标1
                pos_initial=[]
                for i in range(len(sequence)-4):
                    if(sequence[i:i+coding_table_len] in map_rule):
                        pos_initial.append(i)
                pos_initial.append(len(sequence))
                i,j=0,0;separate_initial=[]
                while(i<len(pos_initial)-1):
                    if(i==j):
                        j+=1    
                    elif(pos_initial[j]-pos_initial[i]==coding_table_len):        
                        separate_initial.append([pos_initial[i],pos_initial[j]])
                        i=j
                    elif(pos_initial[j]-pos_initial[i]<coding_table_len):
                        if(j==len(pos_initial)-1):
                            i+=1
                        else:
                            j+=1    
                    elif(pos_initial[j]-pos_initial[i]>coding_table_len):
                        i+=1
            
                merge_init=merge_intervals_by_divide_conquer(separate_initial)
            
                therashold=10
                right_merge=[]
                for i in merge_init:
                    if(i[1]-i[0]>=therashold-1):
                        right_merge.append([i[0],i[1]-1])
                h=subtract_intervals([0,len(sequence)-1],right_merge)
                correct=sequence;
                if(len(h)!=0):   
                #定位错误位置
                    wrong_merge=error_combin_merge(subtract_intervals([0,len(sequence)-1],right_merge))    
                    #区间之间的插入删除
        
                    step=[]
                    for single in wrong_merge:
                        if(len(single)%5==0):
                            step.append([0])
                        elif(len(single)%5==1):
                            step.append([-1,4])
                        elif(len(single)%5==4):
                            step.append([1,-4])
                        elif(len(single)%5==2):
                            step.append([-2,3])
                        elif(len(single)%5==3):
                            step.append([2,-3])  
                    matrix_correct_tmp=[];matrix_correct=[]
                    for length in range(len(step)):
                        matrix_correct_tmp.append(-9999999999999)
                    dfs(0)
                    
                    
                    yy=[]
                    for step_3 in range(len(right_merge)):
                        yy.append(sequence[right_merge[step_3][0]:right_merge[step_3][-1]+1])
                    digital=[]
                    for group in yy:
                        for number in range(0,len(group),5):
                            digital.append(swapped_dict[group[number:number+5]])
                    median=(np.median(np.array(digital)))
                    coding_median=[]

                    for i in range(0,10):
                        coding_median.append(dic[min(int(median+i),255)])
                        coding_median.append(dic[max(0,int(median)-i)])
                    
                    xx=[];minnn=100000;tmp_matrix=""
                    if(len(matrix_correct)==0):
                        print(0/0)
                    for i in matrix_correct:
                        if(minnn>sum(abs(num) for num in i)):
                            minnn=sum(abs(num) for num in i);
                            tmp_matrix=i
                    matrix_correct=tmp_matrix
                    timeout_duration = 1
                    si.signal(si.SIGALRM, handle_timeout)
                    si.alarm(timeout_duration)
                    for step_2 in range(len(matrix_correct)):
                        if(matrix_correct[step_2]==0):
                            xx.append(correct_sub(sequence[wrong_merge[step_2][0]:wrong_merge[step_2][-1]+1]))
                        elif(matrix_correct[step_2]>0):
                            
                            xx.append(correct_ins(sequence[wrong_merge[step_2][0]:wrong_merge[step_2][-1]+1],matrix_correct[step_2]))
                        elif(matrix_correct[step_2]<0):
                            xx.append(correct_dele(sequence[wrong_merge[step_2][0]:wrong_merge[step_2][-1]+1],abs(matrix_correct[step_2])))
                    si.alarm(0)
                    correct=merge_sequence(xx,yy,right_merge[0][0],wrong_merge[0][0])


                new_sequence+=correct;
                
            except Exception as e:

                iiiii-=1

        five_base_group=split_string_by_length(new_sequence+each[-1],5)    
        correct_pixel=[]
        for five_base in five_base_group:
            correct_pixel.append(swapped_dict[five_base])
        correct_pixel_x.append(np.array(correct_pixel,dtype=np.uint8))
                
    reconstructed_image=bit_to_image(image_name,correct_pixel_x[0],correct_pixel_x[1],correct_pixel_x[2])
    reconstructed_image.save(str(error*100)+"_recon.bmp") 
    image=cv2.imread(str(error*100)+"_recon.bmp")
    median = apply_median_filter(image)
    cv2.imwrite(str(error*100)+"_recon_median.bmp", median)
    
    
    
    
    
    bb=image_name
    img = cv2.imread(bb,1)
    b, g, r = cv2.split(img)
    
    cv2.imwrite("Blue.bmp", b)
    cv2.imwrite("Green.bmp", g)
    cv2.imwrite("Red.bmp", r)
    
    bb=str(error*100)+"_recon_median.bmp"
    img = cv2.imread(bb,1)
    b, g, r = cv2.split(img)

    cv2.imwrite("Blue_1.bmp", b)
    cv2.imwrite("Green_1.bmp", g)
    cv2.imwrite("Red_1.bmp", r)
    
    msee=MSE(cv2.imread("Blue.bmp"),cv2.imread('Blue_1.bmp'))+MSE(cv2.imread("Green.bmp"),cv2.imread('Green_1.bmp'))+MSE(cv2.imread("Red.bmp"),cv2.imread('Red_1.bmp'))
    psnrr=psnr(cv2.imread("Blue.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Blue_1.bmp',cv2.IMREAD_GRAYSCALE))+psnr(cv2.imread("Green.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Green_1.bmp',cv2.IMREAD_GRAYSCALE))+psnr(cv2.imread("Red.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Red_1.bmp',cv2.IMREAD_GRAYSCALE))
    ssimm=ssim(cv2.imread("Blue.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Blue_1.bmp',cv2.IMREAD_GRAYSCALE))+ssim(cv2.imread("Green.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Green_1.bmp',cv2.IMREAD_GRAYSCALE))+ssim(cv2.imread("Red.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Red_1.bmp',cv2.IMREAD_GRAYSCALE))
    msssimm=mssim(cv2.imread("Blue.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Blue_1.bmp',cv2.IMREAD_GRAYSCALE))+mssim(cv2.imread("Green.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Green_1.bmp',cv2.IMREAD_GRAYSCALE))+mssim(cv2.imread("Red.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Red_1.bmp',cv2.IMREAD_GRAYSCALE))

    print(str(round(msee/3,3)))
    print(str(round(psnrr/3,3)))
    print(str(round(ssimm/3,3)))
                                
    print(str(round(msssimm/3,3)))




    
 
