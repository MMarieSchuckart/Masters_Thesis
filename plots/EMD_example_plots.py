""" Plot Example for complex signal 
with linear and non-linear oscillatory properties """


#import emd
import numpy as np
#from scipy import signal, ndimage
import matplotlib.pyplot as plt

sample_rate = 1000
seconds = 10
num_samples = sample_rate*seconds

time_vect = np.linspace(0, seconds, num_samples)

freq = 8


""" Plot 1 """
# a complex signal as a combination of a linear and a non-linear one  

# linear signal
x = np.cos(2*np.pi*freq*time_vect)

# Non-linear signal
y = np.cos(2*np.pi*freq*time_vect) + 0.25*np.cos(2*np.pi*freq*2*time_vect-np.pi)

# combine signals to get complex signal
z = x + y

# Quick summary figure
# set plot size
plt.figure(figsize=(8, 4))

# set lines + colors
plt.plot(time_vect[:sample_rate], x[:sample_rate], color='teal')
plt.plot(time_vect[:sample_rate], y[:sample_rate], color = 'indianred')
plt.plot(time_vect[:sample_rate], z[:sample_rate], color = 'grey', linestyle = "dotted")

# set legend
plt.legend(['lineares Signal', 'non-lineares Signal', 'komplexes Signal'], loc = 'upper right')

# label axes
plt.xlabel('Zeit (in s)')
plt.ylabel('Amplitude')


""" Plot 2 """
# how to decompose non-linear signal into two linear ones

# parts of the non-linear signal
y1 = np.cos(2*np.pi*freq*time_vect)
y2 =  0.25*np.cos(2*np.pi*freq*2*time_vect-np.pi)

# non-linear signal
y = np.cos(2*np.pi*freq*time_vect) + 0.25*np.cos(2*np.pi*freq*2*time_vect-np.pi)

# Quick summary figure
# set plot size
plt.figure(figsize=(8, 4))

# set lines + colors
plt.plot(time_vect[:sample_rate], y1[:sample_rate], color='mediumseagreen')
plt.plot(time_vect[:sample_rate], y2[:sample_rate], color = 'steelblue')
plt.plot(time_vect[:sample_rate], y[:sample_rate], color = 'grey', alpha = 0.3, linewidth = 3)

# set legend
plt.legend(['lineares Signal 1', 'lineares Signal 2', 'non-lineares Signal'], loc = 'upper right')

# label axes
plt.xlabel('Zeit (in s)')
plt.ylabel('Amplitude')


