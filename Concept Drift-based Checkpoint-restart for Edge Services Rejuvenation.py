#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
import math
import time
from numpy import *
import pandas as pd
from pylab import *
from sympy import *
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import random
import heapq
from numpy.linalg import norm
from time import time
from sys import stdout
import scipy.stats
import numpy
import time


# In[2]:


data_path='E:/reliability data/sudden drift data.csv'   #突然漂移


# In[8]:


data_path='E:/reliability data/gradual drift data.csv'   #逐渐漂移


# In[13]:


data_path='E:/reliability data/incremental drift data.csv'   #增量漂移


# In[3]:


data_path='E:/reliability data/reoccurring drift data.csv'   #重现漂移


# In[21]:


data_path='E:/reliability data/row reliability data.csv'   #无漂移


# In[22]:


data=pd.read_csv(data_path,header=None)   #通过read_excel()读取文件，内容存在data中


# In[5]:


def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p, M, base=2)+0.5*scipy.stats.entropy(q, M, base=2)


# In[24]:


sum_no_con_dri_js=[]
sum_con_dri_js=[]
sum_mean_run_time=[]
for ec in range(0,2000):
    no_con_dri_js=[]
    con_dri_js=[]
    r_data=data[ec].tolist()
    win_size=30
    windowData = []
    big_Data = []
    windowTime = []
    historydata=[]
    JS_table={}
    inputData=data[0].tolist()
    Js=[]
    k=4
    w=0.2
    a=random.gauss(0,1)
    b=random.uniform(0,w)
    hash_table={}
    his_num=0
    buc_sam_num=[]
    sum_sam_num=0
    a=[]
    b=[]
    sam_index=[]
    sam_index1=[]
    his_v=[]
    sum1=0
    run_time=[]
    for i in range(4):
        a.append(random.gauss(0,1))
        b.append(random.uniform(0,w))
    for q in range(0,9999):
        #time_start = time.time()
        sam_index=[]
        sam_index1=[]
        bucket=[]
        bucketlen=[]
        sum1=0
        buc_sam_num=[]
        sum_sam_num=0
        his_v=[]
        hash_value=[]
        if len(windowData) < win_size:
            windowData.append(r_data[q])
        else:
            #time_start = time.time()
            output=windowData.pop(0)
            historydata.append(output)
            windowData.append(r_data[q+1])
            #print("窗口长度",windowData,len(windowData))
            #hash_value=[]
            #print("随机数",a,b)
            time_start = time.time()  #开始计时
            for s in range(0,4): 
                hash_value.append((output*a[s]+b[s])//w)
            #print("哈希值",hash_value)
            tag=str(hash_value)
            if tag not in hash_table:
                hash_table[tag]=[his_num]
                his_num=his_num+1
            elif tag in hash_table:
                hash_table[tag].append(his_num)
                his_num=his_num+1
        
            for key,value in hash_table.items():
                bucket.append(value)
            #print("哈希表",hash_table)
            #print("桶",bucket)
            if len(historydata) > win_size:
                #time_start = time.time()
                for i in range(len(bucket)):
                    bucketlen.append(len(bucket[i]))
                    buc_sam_num.append(round((len(bucket[i])/len(historydata))*win_size))
                #print("桶长",bucketlen)
                for c in range(len(bucketlen)):
                    sum1+=bucketlen[c]
                #print("##",sum1)
                for j in range(len(buc_sam_num)):
                    sum_sam_num=sum_sam_num+buc_sam_num[j]
                #print("总抽样数",sum_sam_num)
                if sum_sam_num<win_size:
                    sy=win_size-sum_sam_num
                    #print("不够",sy)
                    maxindex=pd.Series(bucketlen).sort_values(ascending = False).index[:sy]
                    for n in maxindex:
                        buc_sam_num[n]+=1
                    '''
                    c=sy//3
                    #print("最大值索引",maxindex)
                    buc_sam_num[maxindex[2]]+=c
                    buc_sam_num[maxindex[1]]+=c
                    buc_sam_num[maxindex[0]]+=(sy-2*c)
                    '''
                if sum_sam_num>win_size:
                    cz=sum_sam_num-win_size
                    #print("多了",cz)
                    maxindex1=pd.Series(bucketlen).sort_values(ascending = False).index[:cz]     
                #print("最大值索引1",maxindex1)
                    for m in maxindex1:
                        buc_sam_num[m]-=1
                    '''
                    d=cz//3
                    buc_sam_num[maxindex1[0]]-=(cz-2*d)
                    buc_sam_num[maxindex1[1]]-=d
                    buc_sam_num[maxindex1[2]]-=d
                    '''
                #print("每个桶抽样个数",buc_sam_num)                    
                for i in range(len(buc_sam_num)):
                    sam_index.append(random.sample(bucket[i],buc_sam_num[i]))
                #print("被抽到的索引号1",sam_index)
                for i in range(len(sam_index)):
                    if sam_index[i]!=[]:
                        for j in sam_index[i]:
                            sam_index1.append(j)
                #print("抽到的索引号2",sam_index1)
                for i in sam_index1:
                    his_v.append(historydata[i])
                his_vector = np.array(his_v)
                #print("***窗口长度",windowData,len(windowData))
                win_v = np.array(windowData)
                #print("历史向量",his_vector,len(his_vector))
                #print("窗口向量",win_v,len(win_v))
                data_js=JS_divergence(his_vector,win_v)
                time_end = time.time()   #结束计时
                #print(q+1,"JS散度",data_js)
                time_c= time_end - time_start
                run_time.append(time_c)
                #print('time cost', time_c, 's')
                JS_table[q+1]=data_js
    #for key in JS_table.keys(): 
    #    print(key,JS_table[key]) 
    #print("运行时间",run_time)
    sum_mean_run_time.append(np.mean(run_time))
    #print("平均运行时间",np.mean(run_time))
    #无漂移数据检测
    for key in JS_table.keys():
            no_con_dri_js.append(JS_table[key])
    sum_no_con_dri_js.append(no_con_dri_js)
    print(ec)
    #漂移数据检测   
    for key in JS_table.keys():
        if key<=9400:
            no_con_dri_js.append(JS_table[key])
        else:
            con_dri_js.append(JS_table[key])
    sum_no_con_dri_js.append(no_con_dri_js)
    sum_con_dri_js.append(con_dri_js)
    print(ec)
    


# ## 突然漂移

# In[6]:


sud_no_cep_dri_max_js=[]
sud_cep_dri_max_js=[]
for i in range(len(sum_no_con_dri_js)):
    sud_no_cep_dri_max_js.append(max(sum_no_con_dri_js[i]))
for i in range(len(sum_con_dri_js)):
    sud_cep_dri_max_js.append(max(sum_con_dri_js[i]))
print("无漂移数据最大散度值",sud_no_cep_dri_max_js)
print("无漂移数据最大散度值",len(sud_no_cep_dri_max_js))
print("突然漂移数据最大散度值",sud_cep_dri_max_js)
print("平均运行时间",np.mean(sum_mean_run_time))


# In[7]:


a=[]
print(max(sud_no_cep_dri_max_js))
for i in sud_cep_dri_max_js:
    if i>=0.2:
        a.append(i)
print(len(a))


# ## 逐渐漂移

# In[19]:


gra_no_cep_dri_max_js=[]
gra_cep_dri_max_js=[]
for i in range(len(sum_no_con_dri_js)):
    gra_no_cep_dri_max_js.append(max(sum_no_con_dri_js[i]))
for i in range(len(sum_con_dri_js)):
    gra_cep_dri_max_js.append(max(sum_con_dri_js[i]))
#print("无漂移数据最大散度值",gra_no_cep_dri_max_js)
#print("无漂移数据最大散度值",len(gra_no_cep_dri_max_js))
print("突然漂移数据最大散度值",gra_cep_dri_max_js)
print("平均运行时间",np.mean(sum_mean_run_time))


# In[21]:


a=[]
print(max(gra_no_cep_dri_max_js))
for i in gra_cep_dri_max_js:
    if i>=0.2:
        a.append(i)
print(len(a))


# ## 增量漂移

# In[34]:


inc_no_cep_dri_max_js=[]
inc_cep_dri_max_js=[]
for i in range(len(sum_no_con_dri_js)):
    inc_no_cep_dri_max_js.append(max(sum_no_con_dri_js[i]))
for i in range(len(sum_con_dri_js)):
    inc_cep_dri_max_js.append(max(sum_con_dri_js[i]))
print("无漂移数据最大散度值",inc_no_cep_dri_max_js)
print("无漂移数据最大散度值",len(inc_no_cep_dri_max_js))
print("突然漂移数据最大散度值",inc_cep_dri_max_js)
print("平均运行时间",np.mean(sum_mean_run_time))

    


# In[1]:


a=[]
print(max(inc_no_cep_dri_max_js))
for i in inc_cep_dri_max_js:
    if i>=0.195:
        a.append(i)
print(len(a))


# In[ ]:


inc = pd.DataFrame(sud_cep_dri_max_js)
inc_cep_dri_max_js.to_csv(r"E:\reliability data\inc_cep_dri_max_js.csv",index=None)


# ## 重现漂移

# In[17]:


reo_no_cep_dri_max_js=[]
reo_cep_dri_max_js=[]
for i in range(len(sum_no_con_dri_js)):
    reo_no_cep_dri_max_js.append(max(sum_no_con_dri_js[i]))
for i in range(len(sum_con_dri_js)):
    reo_cep_dri_max_js.append(max(sum_con_dri_js[i]))
print("无漂移数据最大散度值",reo_no_cep_dri_max_js)
print("无漂移数据最大散度值",len(reo_no_cep_dri_max_js))
print("突然漂移数据最大散度值",reo_cep_dri_max_js)
print("平均运行时间",np.mean(sum_mean_run_time))


# In[19]:


a=[]
print(max(reo_no_cep_dri_max_js))
for i in reo_cep_dri_max_js:
    if i>=0.20:
        a.append(i)
print(len(a))


# In[12]:


js = pd.DataFrame({'sud':sud_cep_dri_max_js,'gra':gra_cep_dri_max_js,'inc':inc_cep_dri_max_js})
js.to_csv(r"E:\reliability data\cep_dri_vec_js.csv",index=None)


# ## 无漂移

# In[27]:


no_cep_dri_max_js=[]
for i in range(len(sum_no_con_dri_js)):
    no_cep_dri_max_js.append(max(sum_no_con_dri_js[i]))
print("无漂移数据最大散度值",no_cep_dri_max_js)
print("平均运行时间",np.mean(sum_mean_run_time))


# In[29]:


js = pd.DataFrame({'no':no_cep_dri_max_js})
js.to_csv(r"E:\reliability data\js.csv",index=None)


# In[ ]:




