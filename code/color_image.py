from PIL import Image
import numpy as np
import itertools
from collections import Counter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as MSE
import cv2
from scipy import signal
from scipy import ndimage
import signal as sig




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
 
def mssim_1(img1, img2):

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



def logistic_map(x, r):
    return r * x*(1-x)

def split_string_by_length(string, length):
    result = []
    for i in range(0, len(string), length):
        result.append(string[i:i+length])
    return result 
def complement_length(array,each_length):
    while(len(array)%(each_length/5)!=0):
        array = np.append(array, 0)
    return array

import random

def mutate(sequence,rate,error_composition):
    sub_rate=error_composition[0]/sum(error_composition)
    ins_rate=error_composition[1]/sum(error_composition)
    del_rate=error_composition[2]/sum(error_composition)
    dna = list(sequence)
    random.seed(None)
    for index, char in enumerate(dna):
        if (random.random() <= rate):
            h=random.random()
            if(h<=sub_rate):

                dna[index]=sub(dna[index])
            elif(h<=(ins_rate+sub_rate)):

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
    
def cal_all_base_group_initial(sequence,each_base_group_length):
    pos_initial=[]
    for i in range(len(sequence)-each_base_group_length-1):
        if(sequence[i:i+each_base_group_length] in map_rule):
            pos_initial.append(i)
    pos_initial.append(len(sequence))
    return pos_initial


def cal_consecutive_merge(pos_initial):
    i,j=0,0;separate_initial=[]
    while(i<len(pos_initial)-1):
        if(i==j):
            j+=1    
        elif(pos_initial[j]-pos_initial[i]==each_base_group_length):        
            separate_initial.append([pos_initial[i],pos_initial[j]])
            i=j
        elif(pos_initial[j]-pos_initial[i]<each_base_group_length):
            if(j==len(pos_initial)-1):
                i+=1
            else:
                j+=1    
        elif(pos_initial[j]-pos_initial[i]>each_base_group_length):
            i+=1

    merge_init=merge_intervals_by_divide_conquer(separate_initial)
    return merge_init
    
def subtract_intervals(big_interval, small_intervals):
    big_list = list(range(big_interval[0], big_interval[1]+1))
    result = [i for i in big_list if all(i < s or i > e for s, e in small_intervals)]
    return result

    
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
def cal_exceed_therashold_merge(merge_init,therashold):
    right_merge=[]
    for each_merge in merge_init:
        if(each_merge[1]-each_merge[0]>=therashold-1):
            right_merge.append([each_merge[0],each_merge[1]-1])
    return right_merge

def optimal_correct_step(wrong_merge):
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
    all_selection = list(itertools.product(*step))
    valid_selection = [selection for selection in all_selection if sum(selection) == 150-len(error_sequence)]
    return valid_selection
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
def hm_distance(str1, str2):
    count = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            count += 1
    return count
    
def correct_sub(wrong):
    xx=""
    for i in range(0,len(wrong),each_base_group_length):
        if(wrong[i:i+each_base_group_length] in map_rule):
            xx+=wrong[i:i+each_base_group_length]
        else:

            minn=10000;tmp_minn=10000;
            for temp in map_rule:
                minn=min(hm_distance(wrong[i:i+each_base_group_length],temp),minn)
                if(minn<tmp_minn):
                    tmp_minn=minn;tmp_xx=temp
            xx+=tmp_xx
    return xx
            
def correct_dele(wrong,num):
    temp_wrong=wrong
    sta=time.time()
    try:
        sig.signal(sig.SIGALRM, handle_timeout)
        sig.setitimer(sig.ITIMER_REAL, 0.1)    
        tmp_corect="";tmp_haiming=10000
        if(num==1):
            for pos_1 in range(len(temp_wrong)):
                wrong=temp_wrong[:pos_1]+temp_wrong[pos_1+1:]
                haiming_1=0;flag=1
                for i in range(0,len(wrong),each_base_group_length):
                    xxxx=wrong[i:i+each_base_group_length]
                    if(gc(xxxx)<0.4 or gc(xxxx)>0.6):
                        flag=0; break
                    if(xxxx in map_rule):
                        haiming_1+=0
                    else:  
                        minn=10000
                        for temp in map_rule:
                            minn=min(hm_distance(xxxx,temp),minn)
                        haiming_1+=minn
                if(haiming_1==0 and flag==1):
                    sig.setitimer(sig.ITIMER_REAL, 0)
                    return wrong
                if(haiming_1<tmp_haiming and flag==1):
                    tmp_haiming=haiming_1;tmp_corect=wrong
        if(num==2):
            for pos_1 in range(len(temp_wrong)):
                t_wrong=temp_wrong[:pos_1]+temp_wrong[pos_1+1:]
                for pos_2 in range(pos_1,len(t_wrong)):
                    wrong=t_wrong[:pos_2]+t_wrong[pos_2+1:]
                    haiming_1=0;flag=1
                    for i in range(0,len(wrong),each_base_group_length):
                        xxxx=wrong[i:i+each_base_group_length]
                        if(gc(xxxx)<0.4 or gc(xxxx)>0.6):
                            flag=0; break
                        if(xxxx in map_rule):
                            haiming_1+=0
                        else:  
                            minn=10000
                            for temp in map_rule:
                                minn=min(hm_distance(xxxx,temp),minn)
                            haiming_1+=minn
                    if(haiming_1==0 and flag==1):
                        sig.setitimer(sig.ITIMER_REAL, 0)
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
                        for i in range(0,len(wrong),each_base_group_length):
                            xxxx=wrong[i:i+each_base_group_length]
                            if(gc(xxxx)<0.4 or gc(xxxx)>0.6):
                                flag=0; break
                            if(xxxx in map_rule):
                                haiming_1+=0
                            else:  
                                minn=10000
                                for temp in map_rule:
                                    minn=min(hm_distance(xxxx,temp),minn)
                                haiming_1+=minn
                        if(haiming_1==0 and flag==1):
                            sig.setitimer(sig.ITIMER_REAL, 0)
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
                            for i in range(0,len(wrong),each_base_group_length):
                                xxxx=wrong[i:i+each_base_group_length]
                                if(gc(xxxx)<0.4 or gc(xxxx)>0.6):
                                    flag=0; break
                                if(xxxx in map_rule):
                                    haiming_1+=0
                                else:  
                                    minn=10000
                                    for temp in map_rule:
                                        minn=min(hm_distance(xxxx,temp),minn)
                                    haiming_1+=minn
                            if(haiming_1==0 and flag==1):
                                sig.setitimer(sig.ITIMER_REAL, 0)
                                return wrong
                            if(haiming_1<tmp_haiming and flag==1):
                                tmp_haiming=haiming_1;tmp_corect=wrong
        sig.setitimer(sig.ITIMER_REAL, 0)
        if(tmp_corect==""):
            return ("NNNNN"*(int((len(temp_wrong)-num)/5)))
        
        return correct_sub(tmp_corect)
    except Exception as e:
        return ("NNNNN"*(int((len(temp_wrong)-num)/5)))
import time
def correct_ins(wrong,num):
    temp_wrong=wrong
    sta=time.time()
    
    try:      
        sig.signal(sig.SIGALRM, handle_timeout)
        sig.setitimer(sig.ITIMER_REAL, 0.1)
        tmp_corect="";tmp_haiming=10000
        if(num==1):
            for pos_1 in range(len(temp_wrong)):
                for base_1 in base:
                    wrong=temp_wrong[:pos_1]+base_1+temp_wrong[pos_1:]
                    if(test(wrong)==False):
                        continue                
                    haiming_1=0;flag=1
                    for i in range(0,len(wrong),each_base_group_length):
                        xxxx=wrong[i:i+each_base_group_length]
                        if(gc(xxxx)<0.4 or gc(xxxx)>0.6):
                            flag=0; break
                        if(xxxx in map_rule):
                            haiming_1+=0
                        else:  
                            minn=10000
                            for temp in map_rule:
                                minn=min(hm_distance(xxxx,temp),minn)
                            haiming_1+=minn
                    if(haiming_1==0 and flag==1):
                        sig.setitimer(sig.ITIMER_REAL, 0)
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
                            for i in range(0,len(wrong),each_base_group_length):
                                xxxx=wrong[i:i+each_base_group_length]
                                if(gc(xxxx)<0.4 or gc(xxxx)>0.6):
                                    flag=0; break
                                if(xxxx in map_rule):
                                    haiming_1+=0
                                else:  
                                    minn=10000
                                    for temp in map_rule:
                                        minn=min(hm_distance(xxxx,temp),minn)
                                    haiming_1+=minn
                            if(haiming_1==0 and flag==1):
                                sig.setitimer(sig.ITIMER_REAL, 0)
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
                                    for i in range(0,len(wrong),each_base_group_length):
                                        xxxx=wrong[i:i+each_base_group_length]
                                        if(gc(xxxx)<0.4 or gc(xxxx)>0.6):
                                            flag=0; break
                                        if(xxxx in map_rule):
                                            haiming_1+=0
                                        else:  
                                            minn=10000
                                            for temp in map_rule:
                                                minn=min(hm_distance(xxxx,temp),minn)
                                            haiming_1+=minn
                                    if(haiming_1==0 and flag==1):
                                        sig.setitimer(sig.ITIMER_REAL, 0)
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
                                            for i in range(0,len(wrong),each_base_group_length):
                                                xxxx=wrong[i:i+each_base_group_length]
                                                if(gc(xxxx)<0.4 or gc(xxxx)>0.6):
                                                    flag=0; break
                                                if(xxxx in map_rule):
                                                    haiming_1+=0
                                                else:  
                                                    minn=10000
                                                    for temp in map_rule:
                                                        minn=min(hm_distance(xxxx,temp),minn)
                                                    haiming_1+=minn
                                            if(haiming_1==0 and flag==1):
                                                sig.setitimer(sig.ITIMER_REAL, 0)
                                                return wrong
                                            if(haiming_1<tmp_haiming and flag==1):
                                                tmp_haiming=haiming_1;tmp_corect=wrong
        sig.setitimer(sig.ITIMER_REAL, 0)
        if(tmp_corect==""):
            return ("NNNNN"*(int((len(temp_wrong)+num)/5)))
        return correct_sub(tmp_corect)
        
    except Exception as e:
        return ("NNNNN"*(int((len(temp_wrong)+num)/5)))
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
def replace_NNNNN_with_median(lst):
    filtered_list = [x for x in lst if x != -1]
    median_pixel=np.median(filtered_list)
    result_list = [x if x != -1 else median_pixel for x in lst]
    
    return result_list
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

def apply_median_filter(image):
    return cv2.medianBlur(image, 3)
def handle_timeout(signum, frame):
    raise Exception("Timeout!")

if __name__ == '__main__':    
    file=open("coding_table.txt")
    map_rule=[]   #construct coding table
    for line in file.readlines():
       map_rule.append(line.rstrip("\n"))   
    file.close()
    image_name= "Lisa.jpg"
    timeout_duration = 0.1;error_composition=[1,1,1]
    red,green,blue=image_to_bitstream(image_name)

    for base_error_rate in [0.01,0.02,0.03,0.04,0.05]:  #set base error rate
        
        correct_pixel=[]
        for all_pixel in [red,green,blue]:
            all_pixel=complement_length(all_pixel,150)
            total_DNAsequence=""

            x= 0.4 
            r = 3.9 
            for pixel in all_pixel:   #chaotic map
                x = logistic_map(x, r) 
                s = int((x*1000))
                new = (pixel+s)%256
                total_DNAsequence+=map_rule[new];
            
            each_DNA=split_string_by_length(total_DNAsequence, 150)
        
            
            each_base_group_length=5;therashold=10;base=["A","C","G","T"];
            all_correct_sequence=[]
            for sequence in each_DNA:
                while(1): 
                    error_sequence=mutate(sequence,base_error_rate,error_composition)
                    all_base_group_initial=cal_all_base_group_initial(error_sequence,each_base_group_length)
                    all_merge=cal_consecutive_merge(all_base_group_initial)
                    right_merge=cal_exceed_therashold_merge(all_merge,therashold)    
                    wrong_merge=error_combin_merge(subtract_intervals([0,len(error_sequence)-1],right_merge))
                    if(wrong_merge!=[]):
                        valid_selection=optimal_correct_step(wrong_merge)
                        if not valid_selection:
                            continue
                        else:
                            min_valid_selection = min(valid_selection, key=lambda x: sum(abs(i) for i in x))
                        yy=[];xx=[]
                        for right in range(len(right_merge)):
                            yy.append(error_sequence[right_merge[right][0]:right_merge[right][-1]+1])

                        for wrong in range(len(min_valid_selection)):
                            if(min_valid_selection[wrong]==0):
                                xx.append(correct_sub(error_sequence[wrong_merge[wrong][0]:wrong_merge[wrong][-1]+1]))
                            elif(min_valid_selection[wrong]>0):
                                
                                xx.append(correct_ins(error_sequence[wrong_merge[wrong][0]:wrong_merge[wrong][-1]+1],min_valid_selection[wrong]))
                            elif(min_valid_selection[wrong]<0):
                                xx.append(correct_dele(error_sequence[wrong_merge[wrong][0]:wrong_merge[wrong][-1]+1],abs(min_valid_selection[wrong])))

                        if(len(right_merge)!=0):
                            correct_sequence=merge_sequence(xx,yy,right_merge[0][0],wrong_merge[0][0])
                        else:
                            correct_sequence="".join(xx)
                        if(correct_sequence=="N"*150):
                            continue
                    else: #no_error or all_base_group of sequence meet map rule
                        correct_sequence=error_sequence
                    all_correct_sequence.append(correct_sequence)
                    break

            swapped_map_rule = {index: value for value, index in enumerate(map_rule)}
            x= 0.4;r = 3.9
            decode_pixel=[]   
            for sequence in all_correct_sequence:
                five_base_group=split_string_by_length(sequence,5)
                each_decode_pixel=[]
                for each in five_base_group:
                    x = logistic_map(x, r) 
                    s = int(x*1000)
                    if(each!="N"*5):
                        temp_pixel=swapped_map_rule[each]-s%256
                        if(temp_pixel<0):
                            each_decode_pixel.append(256+temp_pixel)
                        else:
                            each_decode_pixel.append(temp_pixel)
                    else:
                        each_decode_pixel.append(-1)
                decode_pixel+=replace_NNNNN_with_median(each_decode_pixel)

            
            correct_pixel.append(np.array(decode_pixel[:65536],dtype=np.uint8))
        reconstructed_image=bit_to_image(image_name,correct_pixel[0],correct_pixel[1],correct_pixel[2])
        reconstruct_image_name=image_name+"_"+str(base_error_rate)+"_"+str(error_composition)+".bmp"
        reconstructed_image.save(reconstruct_image_name)
        
        
        
        image=cv2.imread(reconstruct_image_name)
        median = apply_median_filter(image)
        cv2.imwrite("median_fliter_"+reconstruct_image_name, median)
        
        
        #color    
        origin_image = cv2.imread(image_name,1)
        b, g, r = cv2.split(origin_image)
        cv2.imwrite("Blue.bmp", b)
        cv2.imwrite("Green.bmp", g)
        cv2.imwrite("Red.bmp", r)

        print(reconstruct_image_name)
        reconstructed_image_name=reconstruct_image_name
        reconstructed_image = cv2.imread(reconstructed_image_name,1)
        b, g, r = cv2.split(reconstructed_image)
        cv2.imwrite("Blue_1.bmp", b)
        cv2.imwrite("Green_1.bmp", g)
        cv2.imwrite("Red_1.bmp", r)

        msee=MSE(cv2.imread("Blue.bmp"),cv2.imread('Blue_1.bmp'))+MSE(cv2.imread("Green.bmp"),cv2.imread('Green_1.bmp'))+MSE(cv2.imread("Red.bmp"),cv2.imread('Red_1.bmp'))
        psnrr=psnr(cv2.imread("Blue.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Blue_1.bmp',cv2.IMREAD_GRAYSCALE))+psnr(cv2.imread("Green.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Green_1.bmp',cv2.IMREAD_GRAYSCALE))+psnr(cv2.imread("Red.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Red_1.bmp',cv2.IMREAD_GRAYSCALE))
        ssimm=ssim(cv2.imread("Blue.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Blue_1.bmp',cv2.IMREAD_GRAYSCALE))+ssim(cv2.imread("Green.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Green_1.bmp',cv2.IMREAD_GRAYSCALE))+ssim(cv2.imread("Red.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Red_1.bmp',cv2.IMREAD_GRAYSCALE))
        msssimm=mssim_1(cv2.imread("Blue.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Blue_1.bmp',cv2.IMREAD_GRAYSCALE))+mssim_1(cv2.imread("Green.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Green_1.bmp',cv2.IMREAD_GRAYSCALE))+mssim_1(cv2.imread("Red.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Red_1.bmp',cv2.IMREAD_GRAYSCALE))

        
        print(round((msee/3),3))
        print(round((psnrr/3),3))
        print(round((ssimm/3),3))
                                    
        print(round((msssimm/3),3))
        
        
        reconstructed_image_name="median_fliter_"+reconstruct_image_name
        reconstructed_image = cv2.imread(reconstructed_image_name,1)
        b, g, r = cv2.split(reconstructed_image)
        cv2.imwrite("Blue_1.bmp", b)
        cv2.imwrite("Green_1.bmp", g)
        cv2.imwrite("Red_1.bmp", r)

        msee=MSE(cv2.imread("Blue.bmp"),cv2.imread('Blue_1.bmp'))+MSE(cv2.imread("Green.bmp"),cv2.imread('Green_1.bmp'))+MSE(cv2.imread("Red.bmp"),cv2.imread('Red_1.bmp'))
        psnrr=psnr(cv2.imread("Blue.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Blue_1.bmp',cv2.IMREAD_GRAYSCALE))+psnr(cv2.imread("Green.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Green_1.bmp',cv2.IMREAD_GRAYSCALE))+psnr(cv2.imread("Red.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Red_1.bmp',cv2.IMREAD_GRAYSCALE))
        ssimm=ssim(cv2.imread("Blue.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Blue_1.bmp',cv2.IMREAD_GRAYSCALE))+ssim(cv2.imread("Green.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Green_1.bmp',cv2.IMREAD_GRAYSCALE))+ssim(cv2.imread("Red.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Red_1.bmp',cv2.IMREAD_GRAYSCALE))
        msssimm=mssim_1(cv2.imread("Blue.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Blue_1.bmp',cv2.IMREAD_GRAYSCALE))+mssim_1(cv2.imread("Green.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Green_1.bmp',cv2.IMREAD_GRAYSCALE))+mssim_1(cv2.imread("Red.bmp",cv2.IMREAD_GRAYSCALE),cv2.imread('Red_1.bmp',cv2.IMREAD_GRAYSCALE))
        print(reconstruct_image_name)
        print(round((msee/3),3))
        print(round((psnrr/3),3))
        print(round((ssimm/3),3))
                                    
        print(round((msssimm/3),3))

    
