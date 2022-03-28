#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sounddevice as sd
from scipy.io import wavfile
from scipy import signal
#𝑥[𝑛] = 2^𝑛(𝑢[𝑛 + 4] − 𝑢[𝑛 − 3])
#%%
#Zadatak 1
start_n=-4
d_n=1
end_n=3
n=np.arange(start_n, end_n, d_n)
x =(float(2)**n)*((n>=-4)*(n<3))


plt.figure()
plt.stem(n,x)
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('Originalni signal: x[n] = 2^n(u[n + 4] − u[n − 3])')
plt.grid(b=True,which="both",color='grey',linestyle='--')

plt.show()
#%%

#y1[𝑛] = 𝑥[3(𝑄 + 1) − (𝑅 + 1)𝑛] 
#y1[n]=x[3*3-n]
#y1[n]=[9-n ]

#P: 
n0=9

y1 = np.concatenate((x,np.zeros(n0))) 
n1 = np.concatenate((np.arange(start_n-n0, start_n, d_n),n))
plt.figure()
plt.stem(n1,y1)
plt.xlabel('n1')
plt.ylabel('y1[n]')
plt.title('Pomeren signal: y1[n]=x[n+9]')
plt.grid(b=True,which="both",color='grey',linestyle='--')

plt.show()
#%%
#I
y2 = y1[::-1]
n2=-n1[::-1]
plt.figure()
plt.stem(n2,y2)
plt.xlabel('n2')
plt.ylabel('y2[n]')
plt.grid(b=True,which="both",color='grey',linestyle='--')

plt.title('Invertovan signal: y1[n]=x[-n+9]')
plt.show()



#𝑦2[𝑛] = 𝑥[−2(𝑄 + 1) + 𝑛/2].
#y2[n]= x[-2*3+n/2]
#y2[n]=x[-6+n/2] =x[n/2-6]
#%%
#P: 
n0=-6

y1 = np.concatenate((np.zeros(-n0),x)) 
n1 = np.concatenate((n,np.arange(n[-1]+d_n, n[-1]-n0+d_n, d_n)))
plt.figure()
plt.stem(n1,y1)
plt.xlabel('n1')
plt.grid(b=True,which="both",color='grey',linestyle='--')

plt.ylabel('y1[n]')
plt.title('Pomeren signal: y[n]=x[n-6]')
plt.show()
#%%

#S:
skaliranje=1/2  
N = len(y1)
n2=np.arange(n1[0]/skaliranje, n1[-1]/skaliranje+d_n, d_n)
y2 = np.interp(n2, n1/skaliranje, y1)
plt.figure()
plt.stem(n2,y2)
plt.xlabel('n2')
plt.grid(b=True,which="both",color='grey',linestyle='--')

plt.ylabel('y2[n]')
plt.title('Skaliran signal: y[n]=x[n/2-6]')
plt.show()



#%%

# snimanje i analiza zvucnog signala
  # ucestanost odabiranja
duration = 4.7  # trajanje snimka



samplerate, data = wavfile.read('./mic.wav')

dt=1/samplerate
t=np.arange(0,dt*len(data),dt)
chanel1=data # chanel1=data[:,1] ako ima dva kanala
plt.figure()
plt.stem(t,chanel1, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y[t]')
plt.title('Audio signal')
plt.show()

sd.play(data,samplerate)

#%%
#n/2


skaliranje=1/2  
t2=np.arange(t[0]/skaliranje, t[-1]/skaliranje+dt, dt)
y2 = np.interp(t2, t/skaliranje, data)
plt.figure()
plt.stem(t2,y2, markerfmt=" ")
plt.xlabel('t2')
plt.grid(b=True,which="both",color='grey',linestyle='--')

plt.ylabel('y2[t]')
plt.title('Skaliran signal (1/2)')
plt.show()

scaled = np.int16(y2/ np.max (np.abs (y2)) * 32767)
sd.play(scaled, samplerate)

# =============================================================================
# samplerate, data = wavfile.read('./mic.wav')
# samplerate = 22050
# dt=1/samplerate
# t=np.arange(0,dt*len(data),dt)
# chanel1=data # chanel1=data[:,1] ako ima dva kanala
# plt.figure()
# plt.stem(t,chanel1, markerfmt=" ")
# plt.xlabel('t')
# plt.ylabel('y(t)')
# plt.title('Audio signal')
# plt.show()
# 
# sd.play(data,samplerate)
# 
# =============================================================================
#%%
# =============================================================================
# samplerate, data = wavfile.read('./mic.wav')
# samplerate = 88200
# dt=1/samplerate
# t=np.arange(0,dt*len(data),dt)
# chanel1=data # chanel1=data[:,1] ako ima dva kanala
# plt.figure()
# plt.stem(t,chanel1, markerfmt=" ")
# plt.xlabel('t')
# plt.ylabel('y(t)')
# plt.title('Audio signal')
# plt.show()
# 
# sd.play(data,samplerate)
# =============================================================================

#2n


samplerate, data = wavfile.read('./mic.wav')
dt=1/samplerate
t=np.arange(0,dt*len(data),dt)
skaliranje=2

y3 = data[1:-1:skaliranje] # ubrzavanje
t3 = t[1:-1:skaliranje]/skaliranje
plt.figure()
plt.stem(t3,y3, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y[t]')
plt.title('Skalirani signal (2)')
plt.show()
sd.play(y3,samplerate)






#%%

data2=data[::2]
dt=1/22050
t2=np.arange(0,dt*len(data2),dt)

plt.figure()

plt.stem(t2,data2, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y[t]')
plt.title('Audio signal 22kHZ')
plt.show()

scaled = np.int16(data2 / np.max (np.abs (data2)) * 32767)
sd.play(scaled, 22050)

#%%


data3=data2[::2]
dt=1/11025
t3=np.arange(0,dt*len(data3),dt)


plt.figure()

plt.stem(t3,data3, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y[t]')
plt.title('Audio signal 11kHZ')
plt.show()

scaled = np.int16(data3 / np.max (np.abs (data3)) * 32767)
sd.play(scaled, 11025)


#%%

data4=np.zeros(2*len(data2))
i=0
for j in range(len(data2)) :
    data4[i]=data2[j]
    i+=2
   
i=0
for i in range(len(data4)-1): 
    if(i%2):
        data4[i]=(data4[i-1]+data4[i+1])/2
        
        

# skaliranje=1/2
# =============================================================================
# t2=np.arange(t[0]/skaliranje, t[-1]/skaliranje+dt, dt)
# y2 = np.interp(t2, t/skaliranje, data)   
# =============================================================================
dt=1/44100
t4=np.arange(0,dt*len(data4),dt)
plt.figure()

plt.stem(t4,data4, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Audio signal 44kHZ')
plt.show()

scaled = np.int16(data4 / np.max (np.abs (data4)) * 32767)
sd.play(scaled, 44100)


#%%

#Pomeranje za 2 sek

t1=np.concatenate((t,np.arange(t[-1]+dt, t[-1]+2, dt)))
data1=np.concatenate((np.zeros(samplerate*2),data))
plt.figure()
plt.stem(t1,data1, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Audio signal pomeren za 2s')
plt.show()

scaled = np.int16(data1 / np.max (np.abs (data1)) * 32767)
sd.play(scaled, samplerate)

#Invertovanje
#%%

#Inverzija
data2 = data[::-1]


plt.figure()
plt.stem(t,data2, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Audio signal invertovan')
plt.show()

sd.play(data2,samplerate)



#%%
#Palindrom

samplerate, data = wavfile.read('./klav1.wav')
dt=1/samplerate
t=np.arange(0,dt*len(data),dt)

chanel1=data 
plt.figure()
plt.stem(t,chanel1, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Audio signal palindrom')
plt.show()

sd.play(data,samplerate)

#%%

data2 = data[::-1]


plt.figure()
plt.stem(t,data2, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Audio signal palindrom invertovan')
plt.show()

sd.play(data2,samplerate)



#%%
#Zadatak 2
#𝑃 = 1 ℎ[𝑛] = 𝛿[𝑛 − 2] + 𝛿[𝑛]
#𝑥[𝑛] = 2^𝑛(𝑢[𝑛 + 4] − 𝑢[𝑛 − 3])

h=np.zeros(7)
h[4]=h[6]=1

n=np.arange(-4,3,1)
plt.figure()
plt.stem(n,h)
plt.xlabel('n')
plt.ylabel('h[n]')
plt.title('Originalni signal: d[n − 2] + d[n]')
plt.grid(b=True,which="both",color='grey',linestyle='--')

#%%

start_n=-4
d_n=1
end_n=3
n=np.arange(start_n, end_n, d_n)
x =(float(2)**n)*((n>=-4)*(n<3))


plt.figure()
plt.stem(n,x)
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title('Originalni signal: x[n] = 2^n(u[n + 4] − u[n − 3])')
plt.grid(b=True,which="both",color='grey',linestyle='--')

plt.show()

#%%
odziv=np.convolve(h,x)
odziv=odziv/max(np.absolute(odziv))
no=np.arange(-8,5,1)


plt.figure()
plt.stem(no,odziv,markerfmt=" ")
plt.xlabel('n')
plt.ylabel('y[n]')
plt.title('Konvolucija')
plt.grid(b=True,which="both",color='grey',linestyle='--')
plt.show()



#%%


samplerate, data = wavfile.read('./mic.wav')
dt=1/samplerate
t=np.arange(0,dt*len(data),dt)

plt.figure()
plt.stem(t,data, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Audio signal')
plt.show()
#%%
samplerate_impulsni, impulsni_odziv = wavfile.read('./RoomConcertHall.wav')
impulsni_odziv=impulsni_odziv/max(np.absolute(impulsni_odziv))
dt_i=1/samplerate_impulsni
t_i=np.arange(0,dt*len(impulsni_odziv),dt_i)
plt.figure()
plt.stem(t_i,impulsni_odziv,markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Impulsni odziv-RoomConcertHall')
plt.show()
#%%
odziv=np.convolve(data,impulsni_odziv)
odziv=odziv/max(np.absolute(odziv))
dt_o=1/samplerate_impulsni
t_o=np.arange(0,dt*len(odziv),dt_o)
plt.figure()
plt.stem(t_o,odziv,markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Konvolucija-RoomConcertHall')
plt.show()
scaled = np.int16(odziv / np.max (np.abs (odziv)) * 32767)
sd.play(scaled, samplerate)

#%%


samplerate_impulsni, impulsni_odziv = wavfile.read('./RoomHuge.wav')
impulsni_odziv=impulsni_odziv/max(np.absolute(impulsni_odziv))
odziv=np.convolve(data,impulsni_odziv)
odziv=odziv/max(np.absolute(odziv))
dt_o=1/samplerate_impulsni
t_o=np.arange(0,dt*len(odziv),dt_o)
plt.figure()
plt.stem(t_o,odziv,markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Konvolucija-RoomHuge')
plt.show()
scaled = np.int16(odziv / np.max (np.abs (odziv)) * 32767)
sd.play(scaled, samplerate)


#%%

samplerate_impulsni, impulsni_odziv = wavfile.read('./OutdoorBlastoff.wav')
impulsni_odziv=impulsni_odziv/max(np.absolute(impulsni_odziv))
odziv=np.convolve(data,impulsni_odziv)
odziv=odziv/max(np.absolute(odziv))
dt_o=1/samplerate_impulsni
t_o=np.arange(0,dt*len(odziv),dt_o)
plt.figure()
plt.stem(t_o,odziv,markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Konvolucija-OutdoorBlastoff')
plt.show()
scaled = np.int16(odziv / np.max (np.abs (odziv)) * 32767)
sd.play(scaled, samplerate)



#%%

samplerate_impulsni, impulsni_odziv = wavfile.read('./Hangar.wav')
impulsni_odziv=impulsni_odziv/max(np.absolute(impulsni_odziv))
odziv=np.convolve(data,impulsni_odziv)
odziv=odziv/max(np.absolute(odziv))
dt_o=1/samplerate_impulsni
t_o=np.arange(0,dt*len(odziv)-dt_o,dt_o)
plt.figure()
plt.stem(t_o,odziv,markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Konvolucija-Hangar')
plt.show()
scaled = np.int16(odziv / np.max (np.abs (odziv)) * 32767)
sd.play(scaled, samplerate)
#%%
samplerate_impulsni, impulsni_odziv = wavfile.read('./CastleThunder.wav')
impulsni_odziv=impulsni_odziv/max(np.absolute(impulsni_odziv))
odziv=np.convolve(data,impulsni_odziv)
odziv=odziv/max(np.absolute(odziv))
dt_o=1/samplerate_impulsni
t_o=np.arange(0,dt*len(odziv),dt_o)
plt.figure()
plt.stem(t_o,odziv,markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Konvolucija-CastleThunder')
plt.show()
scaled = np.int16(odziv / np.max (np.abs (odziv)) * 32767)
sd.play(scaled, samplerate)


#%%
samplerate_impulsni, impulsni_odziv = wavfile.read('./GiantCave.wav')
impulsni_odziv=impulsni_odziv/max(np.absolute(impulsni_odziv))
odziv=np.convolve(data,impulsni_odziv)
odziv=odziv/max(np.absolute(odziv))
dt_o=1/samplerate_impulsni
t_o=np.arange(0,dt*len(odziv),dt_o)
plt.figure()
plt.stem(t_o,odziv,markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Konvolucija-GiantCave')
plt.show()
scaled = np.int16(odziv / np.max (np.abs (odziv)) * 32767)
sd.play(scaled, samplerate)

#%%
#SLIKA


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = mpimg.imread('./TestSlika.png')
img = rgb2gray(img)
plt.figure()
imgplot = plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()

#%%


M0=np.array([[0, 0, 0],[0, 1, 0],[0, 0, 0]])
img_m0=signal.convolve2d(img, M0, mode='same').clip(0,1)
plt.figure()
imgplot = plt.imshow(img_m0, cmap=plt.get_cmap('gray'))
plt.show()
#%%
M4=np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
img_m4=signal.convolve2d(img, M4, mode='same').clip(0,1)
plt.figure()
imgplot = plt.imshow(img_m4, cmap=plt.get_cmap('gray'))
plt.show()

#%%

M5=np.array([[-1, 0, 1],[2, 0, 2],[-1, 0, 1]])
img_m5=signal.convolve2d(img, M5, mode='same').clip(0,1)
plt.figure()
imgplot = plt.imshow(img_m5, cmap=plt.get_cmap('gray'))
plt.show()


#%%

M7=np.array([[1/16, 2/16, 1/16],[2/16, 4/16, 2/16],[1/16, 2/16, 1/16]])
img_m7=signal.convolve2d(img, M7, mode='same').clip(0,1)
plt.figure()
imgplot = plt.imshow(img_m7, cmap=plt.get_cmap('gray'))
plt.show()

#%%

M8=np.array([[2, 2, -4],[2, -4, 3],[1, 2,-1]])
img_m8=signal.convolve2d(img, M8, mode='same').clip(0,1)
plt.figure()
imgplot = plt.imshow(img_m8, cmap=plt.get_cmap('gray'))
plt.show()



#%%

#Furijeov red



start_t=-4
d_t=0.01
end_t=4
t=np.arange(start_t, end_t, d_t)
#x =(t+2)*((t>-2)*(t<0))+2*(t>=0) *(t<=4)
#V= sin(t)*(u(k*pi)-u(k*pi/2))+(1+cos(t))*(u(pi/2 + k*pi)- u((k+1)*pi))
V=((np.sin(t%np.pi)) *((t%np.pi>=0)*(t%(np.pi)<np.pi/2))) + ((1+np.cos(t%np.pi))*((t%(np.pi) >= np.pi/2)*((t%np.pi)<=np.pi)) )
               
plt.figure()
plt.plot(t,V)
plt.xlabel('t')
plt.ylabel('v(t)')
plt.ylim([0,2])
plt.title('Originalni signal')
plt.show()

#ak= 1/T * int ( V(t) * e^(-jnw0t))
        #%% 
#T=pi
#w0 = 2pi/t = 2
N = 50

pi=np.pi
w0 = 2
k=np.arange(-N,N+1)
A=1j*2*k
Ak= -4*k*k
#ak = 1/pi * ( (-1j*2*np.exp(-1j*k*pi)+1)/ (1-4*k^2) +(1/(2j*k))*(np.exp(-1j*k*pi)-np.exp(-2j*k*pi)) + (2j*k*np.exp(-2j*k*pi)+np.exp(-1j*k*pi))/(1-4*k^2))
ak= 1/pi * ( ((Ak-A+1)*np.exp(-pi*A/2)-np.exp(-pi*A))/(A*Ak+A) + (1-A*np.exp(-pi/2*A))/(Ak+1)   )
#k= 1/pi * (np.exp(-pi*A/2)/(Ak+1) + (np.exp(-pi*A))/(A*Ak+A) + (1-A*np.exp(-pi/2*A))/(Ak+1)   )

ak[N]=2 # a0=0.5 (srednja vrednost signala - DC komponenta)

plt.figure()
plt.stem(k, np.absolute(ak))
plt.xlabel('k')
plt.title('Amplitudski linijski spektar')
plt.show()

plt.figure()
plt.stem(k, np.angle(ak))
plt.xlabel('k')
plt.title('Fazni linijski spektar')
plt.show()
#%%
N = 1
pi=np.pi
w0 = 2
k=np.arange(-N,N+1)
A=1j*2*k
Ak= -4*k*k
ak= 1/pi * ( ((Ak-A+1)*np.exp(-pi*A/2)-np.exp(-pi*A))/(A*Ak+A) + (1-A*np.exp(-pi/2*A))/(Ak+1)   )
# =============================================================================
# ak= 1/pi * (np.exp(-pi*A/2)/(Ak+1) + (np.exp(-pi*A))/(A*Ak+A) + (1-A*np.exp(-pi/2*A))/(Ak+1)   )
# =============================================================================

ak[N]=0.5# a0=0.5 (srednja vrednost signala - DC komponenta)
t=np.arange(-4,4,0.01)

v2=np.zeros(len(t))
for i in range(len(t)):
    v2[i]=np.absolute(ak[N])
    for k in range(1,N+1):
        v2[i] = v2[i] + 2*np.absolute(ak[k+N])*np.cos((k)*w0*t[i]+np.angle(ak[k+N]))   
plt.figure()        
plt.plot(t,V)
plt.ylim([0,2])        
plt.title('K=1');
plt.plot(t,v2)
plt.show()
#%%
N = 2
pi=np.pi
w0 = 2
k=np.arange(-N,N+1)
A=1j*2*k
Ak= -4*k*k
ak= 1/pi * ( ((Ak-A+1)*np.exp(-pi*A/2)-np.exp(-pi*A))/(A*Ak+A) + (1-A*np.exp(-pi/2*A))/(Ak+1)   )
# =============================================================================
# ak= 1/pi * (np.exp(-pi*A/2)/(Ak+1) + (np.exp(-pi*A))/(A*Ak+A) + (1-A*np.exp(-pi/2*A))/(Ak+1)   )
# =============================================================================

ak[N]=0.5# a0=0.5 (srednja vrednost signala - DC komponenta)
t=np.arange(-4,4,0.01)

v2=np.zeros(len(t))
for i in range(len(t)):
    v2[i]=np.absolute(ak[N])
    for k in range(1,N+1):
        v2[i] = v2[i] + 2*np.absolute(ak[k+N])*np.cos((k)*w0*t[i]+np.angle(ak[k+N]))   
plt.figure()        
plt.plot(t,V)
plt.ylim([0,2])      
plt.title('K=2');  
plt.plot(t,v2)
plt.show()
#%%
N = 5
pi=np.pi
w0 = 2
k=np.arange(-N,N+1)
A=1j*2*k
Ak= -4*k*k
ak= 1/pi * ( ((Ak-A+1)*np.exp(-pi*A/2)-np.exp(-pi*A))/(A*Ak+A) + (1-A*np.exp(-pi/2*A))/(Ak+1)   )
# =============================================================================
# ak= 1/pi * (np.exp(-pi*A/2)/(Ak+1) + (np.exp(-pi*A))/(A*Ak+A) + (1-A*np.exp(-pi/2*A))/(Ak+1)   )
# =============================================================================

ak[N]=0.5# a0=0.5 (srednja vrednost signala - DC komponenta)
t=np.arange(-4,4,0.01)

v2=np.zeros(len(t))
for i in range(len(t)):
    v2[i]=np.absolute(ak[N])
    for k in range(1,N+1):
        v2[i] = v2[i] + 2*np.absolute(ak[k+N])*np.cos((k)*w0*t[i]+np.angle(ak[k+N]))   
plt.figure()        
plt.plot(t,V)
plt.ylim([0,2])   
plt.title('K=5');     
plt.plot(t,v2)
plt.show()
#%%
N = 10
pi=np.pi
w0 = 2
k=np.arange(-N,N+1)
A=1j*2*k
Ak= -4*k*k
ak= 1/pi * ( ((Ak-A+1)*np.exp(-pi*A/2)-np.exp(-pi*A))/(A*Ak+A) + (1-A*np.exp(-pi/2*A))/(Ak+1)   )
# =============================================================================
# ak= 1/pi * (np.exp(-pi*A/2)/(Ak+1) + (np.exp(-pi*A))/(A*Ak+A) + (1-A*np.exp(-pi/2*A))/(Ak+1)   )
# =============================================================================

ak[N]=0.5# a0=0.5 (srednja vrednost signala - DC komponenta)
t=np.arange(-4,4,0.01)

v2=np.zeros(len(t))
for i in range(len(t)):
    v2[i]=np.absolute(ak[N])
    for k in range(1,N+1):
        v2[i] = v2[i] + 2*np.absolute(ak[k+N])*np.cos((k)*w0*t[i]+np.angle(ak[k+N]))   
plt.figure()        
plt.plot(t,V)
plt.title('K=10');
plt.ylim([0,2])        
plt.plot(t,v2)
plt.show()
              #%%
N = 50
pi=np.pi
w0 = 2
k=np.arange(-N,N+1)
A=1j*2*k
Ak= -4*k*k
ak= 1/pi * ( ((Ak-A+1)*np.exp(-pi*A/2)-np.exp(-pi*A))/(A*Ak+A) + (1-A*np.exp(-pi/2*A))/(Ak+1)   )
# =============================================================================
# ak= 1/pi * (np.exp(-pi*A/2)/(Ak+1) + (np.exp(-pi*A))/(A*Ak+A) + (1-A*np.exp(-pi/2*A))/(Ak+1)   )
# =============================================================================

ak[N]=0.5# a0=0.5 (srednja vrednost signala - DC komponenta)
t=np.arange(-4,4,0.01)

v2=np.zeros(len(t))
for i in range(len(t)):
    v2[i]=np.absolute(ak[N])
    for k in range(1,N+1):
        v2[i] = v2[i] + 2*np.absolute(ak[k+N])*np.cos((k)*w0*t[i]+np.angle(ak[k+N]))   
plt.figure()        
plt.plot(t,V)
plt.title('K=50');
plt.ylim([0,2])        
plt.plot(t,v2)
plt.show() 