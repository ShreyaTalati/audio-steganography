# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 22:15:06 2018

@author: shreya
"""

#from __future__ import print_function
import scipy.io.wavfile as wavfile
import scipy
from scipy.fftpack import fft

from matplotlib import pyplot as plt
import random
import time
import numpy as np
import cmath
from math import frexp, copysign
from sys import float_info
"""
XOR OF FLOAT NUMBERS
"""
fmax, max_exp, max_10_exp, fmin, min_exp, min_10_exp, dig, mant_dig, epsilon, radix, rounds = float_info

def ifrexp(x):
    """Get the mantissa and exponent of a floating point number as integers."""
    m,e = frexp(x)
    return int(m*2**mant_dig),e


def float_xor(a,b):
    """a ^ b"""
    if a==0.0:
        if copysign(1.0,a)==1.0:
            return b
        else:
            return -b
    if b==0.0:
        return float_xor(b,a)

    if a<0:
        if b<0:
            return float_xor(-a,-b)
        else:
            return -float_xor(-a,b)
    if b<0:
        return -float_xor(a,-b)
            
    if abs(a)>=abs(b):
        return float_xor_(a,b)
    else:
        return float_xor_(b,a)

#The helper functions assume that exponent(a) >= exponent(b).
#The operation lambda x: ~(-x) converts between two's complement and one's complement representation of a negative number. One's complement is more natural for floating point numbers because the zero is signed.

def float_xor_(a,b):
    ma,ea = ifrexp(a)
    mb,eb = ifrexp(b)

    mb = mb>>(ea-eb)

    return ( mb^ma )*2**(ea-mant_dig)

##################################################

#"""
#CONVERT FLOAT TO BINARY
#"""
#def dec2bin(f):
#    if f >= 1:
#        g = int(math.log(f, 2))
#    else:
#        g = -1
#    h = g + 1
#    ig = math.pow(2, g)
#    st = ""    
#    while f > 0 or ig >= 1: 
#        if f < 1:
#            if len(st[h:]) >= 10: # 10 fractional digits max
#                   break
#        if f >= ig:
#            st += "1"
#            f -= ig
#        else:
#            st += "0"
#        ig /= 2
#    st = st[:h] + "." + st[h:]
#    return st

######################################
    
start_time=time.time()
'''
OBTAIN SECRET DATA
'''
#read data to be hidden
f=open("secretData.txt","r")
contents =f.read()
str=''
strn=''
for i in contents:
    str=''
    str+=bin(ord(i))[2:]   #because o/p comes in format - 0b0100...
    if (len(str)!=8):
       for j in range(8-len(str)):
           strn+='0'
       strn+=str
    else:
        strn+=str

a=[]

for i in strn:
    #print(str[i])
    if (i=='0'):
        a.append(-1)
    else:
        a.append(1)
print(a)    



######################################
#read audio file
'''
READ AUDIO FILE
CONVERT OT FREQ. DOMAIN
'''

fs_rate, signal = wavfile.read("input.wav")   #sample rate in smaples/sec and data
print ("Frequency sampling", fs_rate)
#wavfile.write('org_audio.wav',44100,signal)

#signal.shape gives total number of samples and number of channels in a tuple
l_audio = len(signal.shape) 
print ("Channels", l_audio)
if l_audio == 2:
    signal = signal.sum(axis=1) / 2 #obtaining avg of both channels
N = signal.shape[0]
print ("Complete Samplings N", N)
secs = N / float(fs_rate)  #to obtain duration of audio
print ("secs", secs)
Ts = 1.0/fs_rate # sampling interval in time
print ("Timestep between samples Ts", Ts)
t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
#freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
new_sig=[]
for i in signal:
    new_sig.append(i/(pow(2,15)-1))
#print(new_sig)
FFT=np.fft.fft(new_sig)


print("------%s seconds----"%(time.time()-start_time))    

####################################
'''
GENERATE PN SEQ WITH cr=1000
'''
p=[]
i=0
while i<1000:
    # print(i)
    # print(p)
     r=random.randint(-1,2)
     if r==0:
        
         continue
     else:
          p.append(r/abs(r)*1)
          i+=1

print("------%s seconds----"%(time.time()-start_time))          

######################################
'''
MODULATING THE TEXT
'''
alpha=10
b=[]
for i in a:
    for j in p:
        b.append(i*j)
        
w=[]
j=0
for i in range(len(b)):
    if j%len(p)==0:
        j=0
    w.append(alpha*b[i]*p[j])
    j+=1
    

print("------%s seconds----"%(time.time()-start_time))    



######################################

'''
EMBEDDING TEXT
'''
real=[]           #real values of FFT signal
phase={} 
j=0                    
for i in FFT:
    phase[j]=cmath.phase(i)
    real.append(abs(i))
    j+=1

print("LEN OF TEXT", len(a))
print("LEN OF PN SEQ ",len(p))
print("LEN OF KEY1",len(b))
print("LEN OF SIGNAL", len(real))
print("LEN OF MOD. TEXT",len(w))
print("------%s seconds----"%(time.time()-start_time)) 

v=[]
j=0
for i in real:
    if j==len(w):
        j=0
    v.append(float_xor(i,w[j]))
    j+=1
print("LEN OF STEGO AUDIO",len(v))
#binary=[]
#for i in real:
#    binary.append(dec2bin(i))
#print(binary)
#print(real)

print("------%s seconds----"%(time.time()-start_time))    


########################################
'''
ADDING PHASE INFO AFTER EMBEDDING
'''
cmpx=[]
j=0
for i in phase:
    cmpx.append(cmath.rect(v[j],i))
    j+=1
print("------%s seconds----"%(time.time()-start_time))    


print("LEN OF STEGO AUDIO",len(cmpx))  
########################################
'''
WRITING AUDIO FILE
'''
embed_a=np.fft.ifft(cmpx)
n=0
rec=[]
p_rec=[]
for i in embed_a:
#    if i%16==0:
#        n=0
#    n+=i*(2**(15-i))
#    p_rec.append(n)
    rec.append([i.real,i.imag])
    
  
    
rec=np.asarray(rec)
   
scipy.io.wavfile.write('new_audio_emb.wav',fs_rate,rec)



########################################
'''
PLOTTING THE SIGNALS
'''
freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
 
plt.subplot(411)
p1 = plt.plot(t, new_sig, "g") # plotting the original signal(TD)
plt.xlim(0,secs)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig('timeDomain.png',  bbox_inches = 'tight')



plt.subplot(412)
p2 = plt.plot(freqs, FFT, "r") # plotting the complete fft spectrum
plt.xlim((0,10000))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.savefig('freqDomain.png',  bbox_inches = 'tight')


plt.subplot(413)
p2 = plt.plot(freqs, cmpx, "b") # plotting the embedded fft spectrum
plt.xlim((0,10000))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.savefig('embed.png',  bbox_inches = 'tight')


'''
embed_a=np.fft.ifft(cmpx)
#print(embed_a)
plt.subplot(414)
p2 = plt.plot(t,embed_a, "r") # plotting the embedded signal(TD)
plt.xlim(0,secs)
plt.xlabel('Time(s)')
plt.ylabel('Amplitude')
plt.savefig('embed_a.png',  bbox_inches = 'tight')
'''


plt.show()

print("------%s seconds----"%(time.time()-start_time))    

#wavfile.write('embed_audio.wav',441000,embed_a)
print("*********END************")

'''   
pcmpx=[]
extract=[]
j=0
for i in range(len(cmpx)):
    if j%len(p)==0:
        j=0
    pcmpx.append(alpha*cmpx[i]*p[j])
    j+=1
#print(pcmpx)
for i in pcmpx:
    if(i>0):
         extract.append(1)
    else:
         extract.append(-1)

'''

#############################################

'''
EXTRACTION PROCESS
'''
v1=[]
for i in cmpx:
    v1.append(i.real)
   
#w1=[]
#    
#a1=[]
#for i in w1:
#    if(i%len(p)==0):
#        if(i>0):
#            a1.append(1)
#        else:
#            a1.append(0)
w1=[]
s=0
for i in range(len(cmpx)):
    if j%len(p)==0:
        w1.append(s)
        s=0
        j=0
    #w1.append(v1[i]*p[j])
    s+=v[i]*p[j]
    j+=1   

        
b1=[]
for i in range(len(w1)):
    #if(i%len(p)==0):
        if(w1[i]>0):
            b1.append(1)
        else:
            b1.append(0)

c=0
j=1
for i in range(len(a)):
    if a[i]!=b1[j]:
        c+=1
    j+=1
        #print("not equal")
print(c)       
b2=b1[1:]
print(b2)
print("LEN RETRIEVED",len(b2))


get_text=''
n=0
j=1


for i in range(len(b2)):
    if i!=0 and i%8==0:
        #print("FINAL N",n)
        get_text+=chr(int(n))
        print("T ",get_text)
        n=0
        j=1
        
    n+=b2[i]*(2**(8-j))
    #print(n)
    j+=1
    
print()
print(get_text)  
print()
print()
###############################################  
'''
FINDING NOISE
'''
'''
def SNR(a,axis,ddof):        
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

s2n=SNR(FFT,0,0)
s2ne=SNR(cmpx,0,0)
print("NOISE IN ORGINAL SIGNAL",s2n)
print("NOISE IN EMBEEDED SIGNAL",s2ne)
#ratio=s2ne/s2n  
#print(ratio)
'''


'''
#a
#p
#signal         
#b=a*p
#w=alpha*b*p
#v=xor(signal,w)         
#cmpx=v with phase embedded
#pcmpx=p*cmpx         
'''