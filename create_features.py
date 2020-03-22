# -*- coding: utf-8 -*-
"""

@author: Anuradha Kar
"""
###### create_features.py ################
## Python program to generate training features after implementing filtering and
## data augmentation methods on raw gaze data read from CSV files
## data available from: https://data.mendeley.com/datasets/cfm4d9y7bh/1

import pandas as pd
import csv
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage.interpolation import shift
import statsmodels.stats.api as sms


gtx=[]
gty=[]
tim_rel=[]
gaze_gt=[]
gaze_inp=[]
dff=[]
gaze_ang=[]

###########################
# Data label classe numbers: 
#1-> UD50
#2->UD60
#3->UD70
#4->UD80
#5->neutral
#6->roll
#7->pitch
#8->yaw

#Pose labels
#typ-> yaw 20 degrees
#trp-roll 20 degrees
#tpp-pitch 20 degrees
#tnu-neutral 20 degrees
############################
name ="us01_"  ## repeat this routine for all users and all poses
#dist="80"
pose= "typ20"  
label= "8"   
ID1= "Pose_" + pose #dist UDT- user distance tablet/PoseT #expt name
ID2= name   #user name
ID3="gaze"    # category name gaze, yaw, pitch
str1= [label, ID1,ID2,ID3]

#########################################################
#Dataframe with headers F-> feature number
col_names =  ['F1', 'F2', 'F3','F4', 'F5', 'F6','F7', 'F8', 'F9','F10', 'F11', 'F12','F13', 'F14', 'F15','M', 'SD', 'IQR','95U', '95B']
data_df  = pd.DataFrame(columns = col_names)

##########################################################

with open('C:/Users/Documents/Python Scripts/ml_gaze/'+name+pose+'_tab.csv','r') as csvfile:

    datavals = csv.reader(csvfile, delimiter=',')
    datavals.next()
    for r1 in datavals:
        tim_rel.append(float(r1[0]))
        gtx.append(float(r1[1]))
        gty.append(float(r1[2]))
        
        gaze_ang.append(float(r1[14])) #Change column numbers, 14 is gaze frontal angle , 12 pitch, 10 is yaw angle data
        gaze_inp.append(float(r1[14]))
        gaze_gt.append(float(r1[13]))  #Change column numbers, 13 is gaze frontal ground truth , 11 pitch, 9 is yaw angle GT

tim_rel= tim_rel[0:2490]  ##timestamps
        
#### outlier removal and error calculation
gz_filt= signal.medfilt(gaze_ang,41)
for n, i in enumerate(gz_filt):   ###add thresholding
    if i > 50:
        gz_filt[n] = np.median(gz_filt)
gz_err= [abs(m3-n3) for m3,n3 in zip(gz_filt,gaze_gt)]	
##################################################################
###Data augmentations
#################### 1.Mean 2. Horizontal flip 3. vertical flip
aoi_dict={}
mean_err =[]
aoi_std=[]
for m9 in range(0, 28):
    aoi_dict['aoi_%02d' % m9]= gz_err[(m9*89)-30:m9*89]
    lst =  gz_err[(m9*89)-25:(m9*89-5)]
    avg_1= sum(lst)/30
    std_1= np.std(lst)
    mean_err.append(avg_1)  #mean gaze angle differences
    aoi_std.append(std_1)
print len(mean_err)
           
meanvals = [mean_err[0], mean_err[1],mean_err[2],mean_err[5],mean_err[6],
                        mean_err[9],mean_err[10],mean_err[11], mean_err[14],mean_err[15],    ####original data
                        mean_err[19], mean_err[20], mean_err[21],mean_err[24],mean_err[25]]
                        
meanvals_hf= [mean_err[0], mean_err[1],mean_err[2],mean_err[5],mean_err[6],
                        mean_err[19], mean_err[20], mean_err[21],mean_err[24],mean_err[25],   ###aug 1
                        mean_err[9],mean_err[10],mean_err[11], mean_err[14],mean_err[15]]
                        
meanvals_vf= [mean_err[0], mean_err[5],mean_err[6],mean_err[1],mean_err[2],
                        mean_err[9],mean_err[14],mean_err[15], mean_err[10],mean_err[11],
                        mean_err[19], mean_err[24], mean_err[25],mean_err[20],mean_err[21]]   ###aug2
fmean1= np.mean(meanvals)
fstd1=  np.std(meanvals)
conf1= sms.DescrStatsW(gz_err).tconfint_mean(alpha=0.05)
qu, ql= np.percentile(gz_err, [75 ,25])
lst0= [fmean1, fstd1, conf1[0], conf1[1], qu- ql]+str1
f1= meanvals+ lst0
f2= meanvals_hf+lst0
f3= meanvals_vf+lst0 

###############################4. White noise####################################
g_noise= np.random.normal(0, 0.2, len(gz_err))
data_aug4= gz_err+g_noise

aoi_dict4={}
mean4 =[]
for m9 in range(0, 28):
    aoi_dict4['aoi_%02d' % m9]= data_aug4[(m9*89)-30:m9*89]
    lst =  data_aug4[(m9*89)-25:(m9*89-5)]
    avg_4= sum(lst)/30
    mean4.append(avg_4)  #mean gaze angle differences
meanval4 = [mean4[0], mean4[1],mean4[2],mean4[5],mean4[6],
                        mean4[9],mean4[10],mean4[11], mean4[14],mean4[15],    ####original data
                        mean4[19], mean4[20], mean4[21],mean4[24],mean4[25]]
fmean4= np.mean(meanval4)
fstd4=  np.std(meanval4)
conf4= sms.DescrStatsW(data_aug4).tconfint_mean(alpha=0.05)
qu4, ql4= np.percentile(data_aug4, [75 ,25])
lst4= [fmean4, fstd4, conf4[0], conf4[1], qu4-ql4]+ str1
f4= meanval4+ lst4 
#################################5. Pink noise#######################################
def one_over_f(f, knee, alpha):
    desc = np.ones_like(f)
    desc[f<KNEE] = np.abs((f[f<KNEE]/KNEE)**(-alpha))
    desc[0] = 1
    return desc

white_noise_sigma =  0.2 

SFREQ = 2 #Hz
KNEE = 5 / 1e3 #Hz
ALPHA = .7
N = len(gz_err)
wn=np.random.normal(0.,white_noise_sigma*np.sqrt(SFREQ),N)

#shaping in freq domain
s = np.fft.rfft(wn)
f = np.fft.fftfreq(N, d=1./SFREQ)[:len(s)]
f[-1]=np.abs(f[-1])
fft_sim = s * one_over_f(f, KNEE, ALPHA)
T_sim = np.fft.irfft(fft_sim)  #pink noise data

pn= np.append(T_sim, min(T_sim))
pn= pn[0:2490]
data_aug5= gz_err+pn 

aoi_dict5={}
mean5 =[]
for m9 in range(0, 28):
    aoi_dict5['aoi_%02d' % m9]= data_aug5[(m9*89)-30:m9*89]
    lst =  data_aug5[(m9*89)-25:(m9*89-5)]
    avg_5= sum(lst)/30
    mean5.append(avg_5)  #mean gaze angle differences
meanval5 = [mean5[0], mean5[1],mean5[2],mean5[5],mean5[6],
                        mean5[9],mean5[10],mean5[11], mean5[14],mean5[15],    
                        mean5[19], mean5[20], mean5[21],mean5[24],mean5[25]]
fmean5= np.mean(meanval5)
fstd5=  np.std(meanval5)
conf5= sms.DescrStatsW(data_aug5).tconfint_mean(alpha=0.05)
qu5, ql5= np.percentile(data_aug5, [75 ,25])
lst5= [fmean5, fstd5, conf5[0], conf5[1], qu5-ql5]+ str1
f5= meanval5+ lst5
####################################6. Interpolation ###################################
interp = interp1d(tim_rel,gz_err, kind='linear',bounds_error=False, fill_value= np.mean(gz_err))
xnew = np.arange(0, 87150,35)

data_aug6 = interp(xnew)
aoi_dict6={}
mean6 =[]
for m9 in range(0, 28):
    aoi_dict6['aoi_%02d' % m9]= data_aug6[(m9*89)-30:m9*89]
    lst =  data_aug6[(m9*89)-25:(m9*89-5)]
    avg_6= sum(lst)/30
    
    mean6.append(avg_6)  #mean gaze angle differences
meanval6 = [mean6[0], mean6[1],mean6[2],mean6[5],mean6[6],
                        mean6[9],mean6[10],mean6[11], mean6[14],mean6[15],    
                        mean6[19], mean6[20], mean6[21],mean6[24],mean6[25]]
fmean6= np.mean(meanval6)
fstd6=  np.std(meanval6)
conf6= sms.DescrStatsW(data_aug6).tconfint_mean(alpha=0.05)
qu6, ql6= np.percentile(data_aug6, [75 ,25])
lst6= [fmean6, fstd6, conf6[0], conf6[1], qu6-ql6]+ str1
f6= meanval6+ lst6

#################################### 7. Interpolation +white noise 8.interpolation +pink noise
data_aug7 = data_aug6+ g_noise
data_aug8 = data_aug6+ pn

aoi_dict7={}
mean7 =[]
aoi_dict8={}
mean8 =[]


for m9 in range(0, 28):
    aoi_dict7['aoi_%02d' % m9]= data_aug7[(m9*89)-30:m9*89]#: (m9*90)+60]
    aoi_dict8['aoi_%02d' % m9]= data_aug8[(m9*89)-30:m9*89]
    lst7 =  data_aug7[(m9*89)-25:(m9*89-5)]
    lst8 =  data_aug8[(m9*89)-25:(m9*89-5)]
    
    avg_7= sum(lst7)/30
    mean7.append(avg_7)
    avg_8= sum(lst8)/30
    mean8.append(avg_8)

meanval7 = [mean7[0], mean7[1],mean7[2],mean7[5],mean7[6],
                        mean7[9],mean7[10],mean7[11], mean7[14],mean7[15],    
                        mean7[19], mean7[20], mean7[21],mean7[24],mean7[25]]
fmean7= np.mean(meanval7)
fstd7=  np.std(meanval7)
conf7= sms.DescrStatsW(data_aug7).tconfint_mean(alpha=0.05)
qu7, ql7= np.percentile(data_aug7, [75 ,25])
lst7= [fmean7, fstd7, conf7[0], conf7[1], qu7-ql7]+ str1
f7= meanval7+ lst7

####################

meanval8 = [mean8[0], mean8[1],mean8[2],mean8[5],mean8[6],
                        mean8[9],mean8[10],mean8[11], mean8[14],mean8[15],   
                        mean8[19], mean8[20], mean8[21],mean8[24],mean8[25]]
fmean8= np.mean(meanval8)
fstd8=  np.std(meanval8)
conf8= sms.DescrStatsW(data_aug8).tconfint_mean(alpha=0.05)
qu8, ql8= np.percentile(data_aug8, [75 ,25])
lst8= [fmean8, fstd8, conf8[0], conf8[1], qu8-ql8]+ str1
f8= meanval8+ lst8

#########################9. Convolution#########################################################
win = signal.hann(30)
data_aug9 = signal.convolve(gz_err, win, mode="full")/sum(win) 

##################### 10. time warping ################################
val= np.mean(gz_filt[0:10])
xs = np.array(gz_filt)
ys= shift(xs, 10, cval=val)
data_aug10= [abs(m3-n3) for m3,n3 in zip(ys,gaze_gt)]

aoi_dict9={}
mean9 =[]
aoi_dict10={}
mean10 =[]


for m9 in range(0, 28):
    aoi_dict9['aoi_%02d' % m9]= data_aug9[(m9*89)-30:m9*89]
    aoi_dict10['aoi_%02d' % m9]= data_aug10[(m9*89)-30:m9*89]
    lst9 =  data_aug9[(m9*89)-25:(m9*89-5)]
    lst10 =  data_aug10[(m9*89)-25:(m9*89-5)]
    
    avg_9= sum(lst9)/30
    mean9.append(avg_9)
    avg_10= sum(lst10)/30
    mean10.append(avg_10)

meanval9 = [mean9[0], mean9[1],mean9[2],mean9[5],mean9[6],
                        mean9[9],mean9[10],mean9[11], mean9[14],mean9[15],   
                        mean9[19], mean9[20], mean9[21],mean9[24],mean9[25]]
fmean9= np.mean(meanval9)
fstd9=  np.std(meanval9)
conf9= sms.DescrStatsW(data_aug9).tconfint_mean(alpha=0.05)
qu9, ql9= np.percentile(data_aug9, [75 ,25])
lst9= [fmean9, fstd9, conf9[0], conf9[1], qu9-ql9]+ str1
f9= meanval9+ lst9

meanval10 = [mean10[0], mean10[1],mean10[2],mean10[5],mean10[6],
                        mean10[9],mean10[10],mean10[11], mean10[14],mean10[15],    ####original data
                        mean10[19], mean10[20], mean10[21],mean10[24],mean10[25]]
fmean10= np.mean(meanval10)
fstd10=  np.std(meanval10)
conf10= sms.DescrStatsW(data_aug10).tconfint_mean(alpha=0.05)
qu10, ql10= np.percentile(data_aug10, [75 ,25])
lst10= [fmean10, fstd10, conf10[0], conf10[1], qu10-ql10]+ str1
f10= meanval10+ lst10


################# Write all features to file#################
with open("features.csv", "a") as fp1:
    wr = csv.writer(fp1, dialect='excel',lineterminator='\n')
    wr.writerow(f1)
    wr.writerow(f2)
    wr.writerow(f3)
    wr.writerow(f4)
    wr.writerow(f5)
    wr.writerow(f6)
    wr.writerow(f7)
    wr.writerow(f8)
    wr.writerow(f9)
    wr.writerow(f10)

    