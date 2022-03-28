import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sounddevice as sd
import scipy.signal as sci
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import fftshift, fftfreq
from matplotlib.backends.backend_pdf import PdfPages
from scipy.fftpack import fft


pi=np.pi



#%%
#Zad 1

fs=8000
t=np.arange(0,2,1/fs)

y1=5*np.sin(2*pi*100*t)
freq_axis = fftshift(fftfreq(len(y1), 1/fs))

y2=5*np.sin(2*pi*200*t)

plt.figure()
plt.plot(t,y1)
#plt.stem(t,y1, markerfmt=" ")
plt.xlabel('t')
plt.ylabel('y[t]')
plt.xlim((0,0.01))

plt.title('sin 100')
plt.show()
#sd.play(y1,fs)

#%%



X_fft = fft(y1)

Xa = np.abs(X_fft)


X = np.abs(X_fft) # amplitudski spektar
X=X[0:int(np.ceil(len(X)/2))]
freq_axis_positive = np.arange(0, fs/2, fs/len(y1))

plt.figure()
plt.plot(freq_axis_positive,X)
plt.xlabel('w')
plt.ylabel('ak')
plt.title('Amplituda')
plt.xlim((0,500))
plt.show()
#%%


X_fft = fft(y2)
Xa = np.abs(X_fft)



X = np.abs(X_fft) # amplitudski spektar
X=X[0:int(np.ceil(len(X)/2))]
freq_axis_positive = np.arange(0, fs/2, fs/len(y2))

plt.figure()
plt.plot(freq_axis_positive,X)
plt.xlabel('w')
plt.ylabel('ak')
plt.xlim((0,500))
plt.title('Amplituda')
plt.show()

#%%


plt.figure()
plt.plot(t,y2)
plt.xlabel('t')
plt.ylabel('y[t]')
plt.title('sin 200')
plt.xlim((0,0.01))
plt.show()

freq_axis = fftshift(fftfreq(len(y2), 1/fs))
#sd.play(y2,fs)





#%%
#b)

#NE RUNUJ OPET!!!!!!
duration=2
samplerate=8000
myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
sd.wait()  # ceka dok se zavrsi snimanje
wavfile.write('sekvenca.wav', samplerate, myrecording)  # cuva se fajl po odgovarajucim imenom

#%%
samplerate, data = wavfile.read('./sekvenca.wav')
dt=1/samplerate
t=np.arange(0,dt*len(data),dt)
chanel1=data # chanel1=data[:,1] ako ima dva kanala
plt.figure()
plt.plot(t,chanel1)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Slovo A')
plt.show()




#%%
scaled = np.int16(data/ np.max (np.abs (data)) * 32767)
#sd.play(scaled, samplerate)
poc=np.where(t==1.0)[0][0]
kraj=np.where(t==1.05)[0][0]
t2=np.arange(0,0.05,dt)
data2=data[poc:kraj]

plt.figure()
plt.plot(t2,data2)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Slovo A')
plt.show()

#%%
freq_axis = fftshift(fftfreq(len(data2), 1/samplerate))

X_fft = fft(data2)
Xa = np.abs(X_fft)


X = np.abs(X_fft) # amplitudski spektar
X=X[0:int(np.ceil(len(X)/2))]
freq_axis_positive = np.arange(0, samplerate/2, samplerate/len(data2))

plt.figure()
plt.stem(freq_axis_positive,X, markerfmt=" ")
plt.xlabel('w')
plt.ylabel('ak')
plt.title('Amplitudska karakteristika')
plt.show()

#%%
f, t, Sxx = signal.spectrogram(data, samplerate)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

plt.show()


#%%

#2. ZAD

samplerate, data = wavfile.read('./20200071.wav')

dt=1/samplerate
t=np.arange(0,dt*len(data),dt)

plt.figure()
plt.stem(t,data, markerfmt=" ")
plt.xlim((1,1.05))
plt.xlabel('t')
plt.ylabel('y[t]')
plt.title('Audio signal')
plt.show()
#%%
sd.play(data,samplerate)
#%%
freq_axis = fftshift(fftfreq(len(data), 1/samplerate))

X_fft = fft(data)
Xa = np.abs(X_fft)


X = np.abs(X_fft) # amplitudski spektar
X=X[0:int(np.ceil(len(X)/2))]
freq_axis_positive = np.arange(0, samplerate/2, samplerate/len(data))

plt.figure()
plt.xlim((0,2000))
plt.plot(freq_axis_positive,X)
plt.xlabel('w')
plt.ylabel('ak')
plt.title('Frekvencijski oblik')
plt.show()
#%%

f, t, Sxx = signal.spectrogram(data, samplerate)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim((0,2000))
plt.show()


#%%
#b)

#NE RUNUJ OPET!!!!!!
duration=5
samplerate=16000
myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
sd.wait()  # ceka dok se zavrsi snimanje
wavfile.write('gitara.wav', samplerate, myrecording)  # cuva se fajl po odgovarajucim imenom


#%%

samplerate, data = wavfile.read('./gitara.wav')
#data=30000*data
dt=1/samplerate
t=np.arange(0,dt*len(data),dt)
chanel1=data # chanel1=data[:,1] ako ima dva kanala
plt.figure()
plt.plot(t,chanel1)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Audio signal gitara')
plt.show()

scaled = np.int16(data/ np.max (np.abs (data)) * 32767)
sd.play(scaled, samplerate)

X_fft = fft(data)
Xa = np.abs(X_fft)

fs=samplerate
X = np.abs(X_fft) # amplitudski spektar
X=X[0:int(np.ceil(len(X)/2))]
freq_axis_positive = np.arange(0, fs/2, fs/len(data))

plt.figure()
plt.plot(freq_axis_positive,X)
plt.xlabel('w')
plt.ylabel('ak')

plt.title('Amplituda')
plt.show()
#%%
f, t, Sxx = signal.spectrogram(data, samplerate)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim((0,1000))
plt.show()
#%%
#NE RUNUJ OPET!!!!!!!!!!!!!!!!!!!!!!
duration=5
samplerate=44100
myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
sd.wait()  # ceka dok se zavrsi snimanje
wavfile.write('gitarabolja.wav', samplerate, myrecording)  # cuva se fajl po odgovarajucim imenom


#%%

samplerate, data = wavfile.read('./gitaranote44k.wav')

dt=1/samplerate
t=np.arange(0,dt*len(data),dt)
chanel1=data # chanel1=data[:,1] ako ima dva kanala
plt.figure()
plt.plot(t,chanel1)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Audio signal Gitara note')
plt.show()

sd.play(data,samplerate)


#%%
X_fft = fft(data)
Xa = np.abs(X_fft)

fs=samplerate
X = np.abs(X_fft) # amplitudski spektar
X=X[0:int(np.ceil(len(X)/2))]
freq_axis_positive = np.arange(0, fs/2, fs/len(data))

plt.figure()
plt.plot(freq_axis_positive,X)
plt.xlabel('w')
plt.ylabel('ak')

plt.title('Amplituda')
plt.show()
#%%

f, t, Sxx = signal.spectrogram(data, samplerate)
plt.figure(figsize = (8,6))
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim((0,1500))
plt.show()

#%%
#NERUNUJJ!!!!!!!!!!!!
duration=5
samplerate=16000
myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
sd.wait()  # ceka dok se zavrsi snimanje
wavfile.write('gitaranote.wav', samplerate, myrecording)  # cuva se fajl po odgovarajucim imenom



#%%
#NERUNUJJJ
duration=5
samplerate=44100
myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
sd.wait()  # ceka dok se zavrsi snimanje
wavfile.write('gitaranote44k.wav', samplerate, myrecording)  # cuva se fajl po odgovarajucim imenom
#%%
#3. zad




def lowpass(order, sr, freq, sig):
    b, a = sci.butter(order, freq/(sr/2))
    return sci.filtfilt(b,a,sig)


def bandpass(order, sr, f1, f2, sig):
    b, a = sci.butter(order, np.array( ( f1/(sr/2) , f2/(sr/2) ) ) ,btype='bandpass')
    return sci.filtfilt(b,a,sig)


def graphFreq(sig, sr, filename, xlim1, xlim2):
    f = fftshift(fftfreq(len(sig), 1/sr))
    plt.figure(figsize=(10,5))
    plt.xlim(xlim1,xlim2)
    plt.plot(f, fftshift(np.abs(fft(sig))))
    plt.savefig(filename)
    plt.close()

def graphWave(sig, sr, filename):
    t = np.arange(0, len(sig)/sr, 1/sr)
    plt.figure()
    plt.plot(t, sig)
    plt.savefig(filename)
    plt.close()

#%%
#NERUNUJJJ
duration=4
samplerate=8000
myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
sd.wait()  # ceka dok se zavrsi snimanje
wavfile.write('y1.wav', samplerate, myrecording)
#%%
#NERUNUJJJ
duration=4
samplerate=8000
myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
sd.wait()  # ceka dok se zavrsi snimanje
wavfile.write('y2.wav', samplerate, myrecording)
#%% Projektovanje filtra i filtriranje signala


sampleRate1, y1 = wavfile.read('y1.wav')
sampleRate2, y2 = wavfile.read('y2.wav')


#%%
sd.play(y1,sampleRate1)
#%%
sd.play(y2,sampleRate2)
#%%
dt1=1/sampleRate1
y1=y1[:len(y2)]
t1=np.arange(0,dt1*len(y1),dt1)

plt.figure()
plt.plot(t1,y1)
plt.xlabel('t')
plt.ylabel('y1(t)')
plt.title('Audio signal 1')
plt.show()

#%%
dt2=1/sampleRate2
t2=np.arange(0,dt2*len(y2),dt2)
plt.figure()
plt.plot(t2,y2)
plt.xlabel('t')
plt.ylabel('y2(t)')
plt.title('Audio signal 2')
plt.show()

#%% 
X_fft = fft(y1)
#Xa = np.abs(X_fft)

fs=sampleRate1
freq_axis = fftshift(fftfreq(len(y1), 1/fs))

B = np.abs(X_fft) # amplitudski spektar
X=np.concatenate([B[int(np.ceil(len(B))/2):len(B)],B[0:int(np.ceil(len(B)/2))]])

#X=X[0:int(np.ceil(len(X)/2))]
#req_axis_positive = np.arange(0, fs/2, fs/len(y1))

plt.figure()
plt.plot(freq_axis,X)
plt.xlabel('w')
plt.ylabel('ak')
plt.title('Amplituda1')


plt.show()

#%% 
X_fft = fft(y2)
#Xa = np.abs(X_fft)

fs=sampleRate1
freq_axis = fftshift(fftfreq(len(y2), 1/fs))

B = np.abs(X_fft) # amplitudski spektar
X=np.concatenate([B[int(np.ceil(len(B))/2):len(B)],B[0:int(np.ceil(len(B)/2))]])

#X=X[0:int(np.ceil(len(X)/2))]
#freq_axis_positive = np.arange(0, fs/2, fs/len(y1))

plt.figure()
plt.plot(freq_axis,X)
plt.xlabel('w')
plt.ylabel('ak')
#plt.xlim((-30000,30000))
plt.title('Amplituda2')


plt.show()

#%%

# Konstante

freqLP1 = 4000
freqLP2 = 6000
freqChannel = 18000
freqCarrier = 10000
freqBP_l = freqCarrier - freqLP2
freqBP_h = freqCarrier + freqLP2




#%%
# Ulazno filtriranje
y1n = lowpass(10, sampleRate1, freqLP1, y1)
y2n = lowpass(10, sampleRate1, freqLP2, y2)




#%%
dt1=1/sampleRate1
y1=y1[:int(sampleRate1*4.7)]
t1=np.arange(0,dt1*len(y1n),dt1)
plt.figure()
plt.plot(t1,y1n)
plt.xlabel('t')
plt.ylabel('y1n(t)')
plt.title('Audio signal 1')
plt.show()

#%%
dt2=1/sampleRate2
t2=np.arange(0,dt2*len(y2n),dt2)
plt.figure()
plt.plot(t2,y2n)
plt.xlabel('t')
plt.ylabel('y2n(t)')
plt.title('Audio signal 2')
plt.show()

#%% 
X_fft = fft(y1n)
#Xa = np.abs(X_fft)

fs=sampleRate1
freq_axis = fftshift(fftfreq(len(y1n), 1/fs))

B = np.abs(X_fft) # amplitudski spektar
X=np.concatenate([B[int(np.ceil(len(B))/2):len(B)],B[0:int(np.ceil(len(B)/2))]])

#X=X[0:int(np.ceil(len(X)/2))]
#freq_axis_positive = np.arange(0, fs/2, fs/len(y1))

plt.figure()
plt.plot(freq_axis,X)
plt.xlabel('w')
plt.ylabel('ak')
plt.title('Amplituda1n')


plt.show()

#%% 
X_fft = fft(y2n)
#Xa = np.abs(X_fft)

fs=sampleRate1
freq_axis = fftshift(fftfreq(len(y2n), 1/fs))

B = np.abs(X_fft) # amplitudski spektar
X=np.concatenate([B[int(np.ceil(len(B))/2):len(B)],B[0:int(np.ceil(len(B)/2))]])

#X=X[0:int(np.ceil(len(X)/2))]
#freq_axis_positive = np.arange(0, fs/2, fs/len(y1))

plt.figure()
plt.plot(freq_axis,X)
plt.xlabel('w')
plt.ylabel('ak')
plt.title('Amplituda2n')


plt.show()

#%%

wavfile.write('y1n.wav', sampleRate1, np.int16(y1 / np.max(np.abs(y1)) * 32767))
wavfile.write('y2n.wav', sampleRate1, np.int16(y1 / np.max(np.abs(y1)) * 32767))

scaled = np.int16(y1n/ np.max (np.abs (y1n)) * 32767)
sd.play(scaled, sampleRate1)



#%%
scaled = np.int16(y2n/ np.max (np.abs (y2n)) * 32767)
sd.play(scaled, sampleRate1)
#%%
# Modulacija
freqCarrier=10000
t = np.linspace(0, len(y2n)/sampleRate2, num=len(y2n) )
y2m = y2n * np.cos(2*np.pi*freqCarrier*t)


#dt2=1/sampleRate2
#t2=np.arange(0,dt2*len(y2n),dt2)
plt.figure()
plt.plot(t2,y2m)
plt.xlabel('t')
plt.ylabel('y2m(t)')
plt.title('Audio signal 2')
plt.show()



X_fft = fft(y2m)
#Xa = np.abs(X_fft)


fs=sampleRate1

freq_axis = fftshift(fftfreq(len(y2m), 1/fs))

B = np.abs(X_fft) # amplitudski spektar
X=np.concatenate([B[int(np.ceil(len(B))/2):len(B)],B[0:int(np.ceil(len(B)/2))]])

#X=X[0:int(np.ceil(len(X)/2))]
#freq_axis_positive = np.arange(0, fs/2, fs/len(y1))

plt.figure()
plt.plot(freq_axis,X)

plt.xlabel('w')
plt.ylabel('ak')
plt.title('Amplituda2m')


plt.show()

#%%

scaled = np.int16(y2m/ np.max (np.abs (y2m)) * 32767)
sd.play(scaled, sampleRate1)
#%%

# Spajanje
yt = y1n + y2m



#%%
#dt2=1/sampleRate2
#t2=np.arange(0,dt2*len(y2n),dt2)
plt.figure()
plt.plot(t2,yt)
plt.xlabel('t')
plt.ylabel('yt(t)')
plt.title('Audio signal t')
plt.show()

#%%
X_fft = fft(yt)
#Xa = np.abs(X_fft)


fs=sampleRate1
freq_axis = fftshift(fftfreq(len(yt), 1/fs))

B = np.abs(X_fft) # amplitudski spektar
X=np.concatenate([B[int(np.ceil(len(B))/2):len(B)],B[0:int(np.ceil(len(B)/2))]])

#X=X[0:int(np.ceil(len(X)/2))]
#freq_axis_positive = np.arange(0, fs/2, fs/len(y1))

plt.figure()
plt.plot(freq_axis,X)
plt.xlabel('w')
plt.ylabel('ak')
plt.title('Amplitudat')


plt.show()
#%%

scaled = np.int16(yt/ np.max (np.abs (yt)) * 32767)
sd.play(scaled, sampleRate1)
#%%
# Prolaz kroz kanal
yr = lowpass(5, sampleRate1, freqChannel, yt)

#%%
#dt2=1/sampleRate2
#t2=np.arange(0,dt2*len(y2n),dt2)
plt.figure()
plt.plot(t2,yr)
plt.xlabel('t')

plt.ylabel('yr(t)')
plt.title('Audio signal 2')
plt.show()

#%%

X_fft = fft(yr)
#Xa = np.abs(X_fft)


fs=sampleRate1
freq_axis = fftshift(fftfreq(len(yr), 1/fs))

B = np.abs(X_fft) # amplitudski spektar
X=np.concatenate([B[int(np.ceil(len(B))/2):len(B)],B[0:int(np.ceil(len(B)/2))]])

#X=X[0:int(np.ceil(len(X)/2))]
#freq_axis_positive = np.arange(0, fs/2, fs/len(y1))

plt.figure()
plt.plot(freq_axis,X)
plt.xlabel('w')
plt.ylabel('ak')
plt.title('Amplitudar')


plt.show()
#%%

scaled = np.int16(yr/ np.max (np.abs (yr)) * 32767)
sd.play(scaled, sampleRate1)

#%%
# Odvajanje
y1r = lowpass(10, sampleRate1, freqLP1, yr)

y2b = bandpass(10, sampleRate1, freqBP_l,freqBP_h, yr)



y2d = y2b * np.cos(2*np.pi*freqCarrier*t)

y2r = lowpass(10, sampleRate1, freqLP2, y2d)
#%%
#dt2=1/sampleRate2
#t2=np.arange(0,dt2*len(y2n),dt2)
plt.figure()
plt.plot(t2,y1r)
plt.xlabel('t')

plt.ylabel('y1r(t)')
plt.title('Audio signal 1r')
plt.show()

#%%

X_fft = fft(y1r)
#Xa = np.abs(X_fft)


fs=sampleRate1
freq_axis = fftshift(fftfreq(len(y1r), 1/fs))

B = np.abs(X_fft) # amplitudski spektar
X=np.concatenate([B[int(np.ceil(len(B))/2):len(B)],B[0:int(np.ceil(len(B)/2))]])

#X=X[0:int(np.ceil(len(X)/2))]
#freq_axis_positive = np.arange(0, fs/2, fs/len(y1))

plt.figure()
plt.plot(freq_axis,X)
plt.xlabel('w')
plt.ylabel('ak')
plt.title('Amplitudar')


plt.show()

#%%
#dt2=1/sampleRate2
#t2=np.arange(0,dt2*len(y2n),dt2)
plt.figure()
plt.plot(t2,y2b)
plt.xlabel('t')

plt.ylabel('y2b(t)')
plt.title('Audio signal 2b')
plt.show()

#%%

X_fft = fft(y2b)
#Xa = np.abs(X_fft)


fs=sampleRate1
freq_axis = fftshift(fftfreq(len(y2b), 1/fs))

B = np.abs(X_fft) # amplitudski spektar
X=np.concatenate([B[int(np.ceil(len(B))/2):len(B)],B[0:int(np.ceil(len(B)/2))]])

#X=X[0:int(np.ceil(len(X)/2))]
#freq_axis_positive = np.arange(0, fs/2, fs/len(y1))

plt.figure()
plt.plot(freq_axis,X)
plt.xlabel('w')
plt.ylabel('ak')
plt.title('Amplitudar')


plt.show()

#%%
#dt2=1/sampleRate2
#t2=np.arange(0,dt2*len(y2n),dt2)
plt.figure()
plt.plot(t2,y2d)
plt.xlabel('t')

plt.ylabel('y2d(t)')
plt.title('Audio signal 2d')
plt.show()

#%%

X_fft = fft(y2d)
#Xa = np.abs(X_fft)


fs=sampleRate1
freq_axis = fftshift(fftfreq(len(y2d), 1/fs))

B = np.abs(X_fft) # amplitudski spektar
X=np.concatenate([B[int(np.ceil(len(B))/2):len(B)],B[0:int(np.ceil(len(B)/2))]])

#X=X[0:int(np.ceil(len(X)/2))]
#freq_axis_positive = np.arange(0, fs/2, fs/len(y1))

plt.figure()
plt.plot(freq_axis,X)
plt.xlabel('w')
plt.ylabel('ak')
plt.title('Amplitudar')


plt.show()
#%%
#dt2=1/sampleRate2
#t2=np.arange(0,dt2*len(y2n),dt2)
plt.figure()
plt.plot(t2,y2r)
plt.xlabel('t')

plt.ylabel('y2d(t)')
plt.title('Audio signal 2r')
plt.show()

#%%

X_fft = fft(y2r)
#Xa = np.abs(X_fft)


fs=sampleRate1
freq_axis = fftshift(fftfreq(len(y2r), 1/fs))

B = np.abs(X_fft) # amplitudski spektar
X=np.concatenate([B[int(np.ceil(len(B))/2):len(B)],B[0:int(np.ceil(len(B)/2))]])

#X=X[0:int(np.ceil(len(X)/2))]
#freq_axis_positive = np.arange(0, fs/2, fs/len(y1))

plt.figure()
plt.plot(freq_axis,X)
plt.xlabel('w')
plt.ylabel('ak')
plt.title('Amplitudar')


plt.show()
#%%

scaled = np.int16(y2b/ np.max (np.abs (y2b)) * 32767)
sd.play(scaled, sampleRate1)

#%%

scaled = np.int16(y2d/ np.max (np.abs (y2d)) * 32767)
sd.play(scaled, sampleRate1)


#%%


scaled = np.int16(y2d/ np.max (np.abs (y2d)) * 32767)
sd.play(scaled, sampleRate1)
#%%

scaled = np.int16(y1r/ np.max (np.abs (y1r)) * 32767)
sd.play(scaled, sampleRate1)
#%%

scaled = np.int16(y2r/ np.max (np.abs (y2r)) * 32767)
sd.play(scaled, sampleRate1)