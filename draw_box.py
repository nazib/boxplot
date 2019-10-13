import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from itertools import chain
from xml.dom import minidom

df1 = pd.read_csv('InversNet_delni_S018000_INV.csv')
df2 = pd.read_csv('Voxelmorph_MRI_delni_S018000_VM.csv')
df3 = pd.read_csv('all_ANTS.csv')

cdf = pd.concat([df1, df2, df3])    
#mdf = pd.melt(cdf, id_vars=str(df1.columns), var_name=['VM'])
#ax = sns.boxplot(x="Brain Regions", y="Dice", hue="Letter", data=mdf)  
#plt.show()
num = 16
m=0
df =np.zeros((df1.shape[0],num*3))
x = np.arange(0,num*4,4) 
y = np.arange(1,num*4,4) 
z = np.arange(2,num*4,4)
x_axis = []

label_file = "lpba40.label.xml"
label_xml = minidom.parse(label_file)
color_labels = label_xml.getElementsByTagName('label')
regions = [k.getAttribute('fullname').encode('utf-8') for k in color_labels]
regions.pop(0)
regions = list(regions[i] for i in [0,3,9,13,14,16,21,22,24,25,27,29,32,36,44,46])
x_label = ['I','V','S',' ']*num

for i in range(df1.shape[1]/num):
    #bp1 = df1.iloc[:,m:m+num]
    #bp2 = df2.iloc[:,m:m+num]
    #bp3 = df3.iloc[:,m:m+num]
    
    bp1 = df1.iloc[:,[0,3,9,13,14,16,21,22,24,25,27,29,32,36,44,46]]
    bp2 = df2.iloc[:,[0,3,9,13,14,16,21,22,24,25,27,29,32,36,44,46]]
    bp3 = df2.iloc[:,[0,3,9,13,14,16,21,22,24,25,27,29,32,36,44,46]]

    fig, ax= plt.subplots(figsize=(60,36))
       
    p1 = ax.boxplot(bp1.values, positions=x.tolist(), notch=False, widths=0.8, 
                 patch_artist=True, boxprops=dict(facecolor="C1"))
    p2 = ax.boxplot(bp2.values, positions=y.tolist(), notch=False, widths=0.8, 
                 patch_artist=True, boxprops=dict(facecolor="C8"))
    p3 = ax.boxplot(bp3.values, positions=z.tolist(), notch=False, widths=0.8, 
                 patch_artist=True, boxprops=dict(facecolor="C7"))

    ax.legend([p1["boxes"][0], p2["boxes"][0], p3["boxes"][0]], ['Inv', 'VM','SyN'], loc='lower left', fontsize=45)
     
    ax.set_xticklabels(x_label)
    ax.set_xlim(-1,num*4)
    ax.set_ylim(0.2,0.85)
    ax.xaxis.set_ticks(np.arange(0,num*4,1))
    ax.grid(which='major',linestyle="-", linewidth='0.2', color='black')
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(40)
    
    for l in range(num):
        k = [np.mean(bp1.values[:,l]),np.mean(bp2.values[:,l]),np.mean(bp3.values[:,l])]
        if k.index(np.max(k)) == 0:
           ax.get_xticklabels()[x[l]].set_color('red')
        if k.index(np.max(k)) == 1:
           ax.get_xticklabels()[y[l]].set_color('red')
        if k.index(np.max(k)) == 2:
           ax.get_xticklabels()[z[l]].set_color('red')
        
    ax2 = ax.twiny()
    ax2.set_xlim(-1,num*4)
    ax2.set_xticks(y)
    ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
    
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 37))
    ax2.set_xlabel('Regions')

    #xtickNames = plt.setp(ax2, xticklabels=regions[m:m+num])
    xtickNames = plt.setp(ax2, xticklabels=regions)
    plt.setp(xtickNames, rotation=60, fontsize=36)
    fig.savefig("Boxplot{0}.pdf".format(i))
    m+=num

'''
    i=0
    for f,k,t in zip(x,y,z):
        #df[:,k:k+3] = np.asarray([bp1.iloc[:,x[f]],bp2.iloc[:,x[f]],bp3.iloc[:,x[f]]]).T
        df[:,f] = np.asarray(bp1.iloc[:,i]).T
        df[:,k] = np.asarray(bp2.iloc[:,i]).T
        df[:,t] = np.asarray(bp3.iloc[:,i]).T
        x_axis.append(['I','V','S'])
        i+=1
    
    cols = list(chain.from_iterable(x_axis))      
    dp = pd.DataFrame(df,columns=cols)
    myFig = plt.figure()
    bp = dp.boxplot(column=regions[m:num*3])
    #sn = sns.boxplot(x="Brain Regions", y="Dice", hue="Letter", data=bp)     
    myFig.savefig("image_{0}_{1}t.svg".format(m,m+num), format="svg")
    #dp.to_csv("data_{0}_{1}t.csv".format(m,m+num))
    m+=num
    x_axis =[]

bp = df.iloc[:,19:36].boxplot()
myFig.savefig("19_36.svg", format="svg")
bp = df.iloc[:,36:54].boxplot()
myFig.savefig("36_54.svg", format="svg")
'''
