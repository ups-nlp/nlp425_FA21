#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 13:15:14 2021

@author: prowe
"""


# Plot the inputs to see what they look like

ifig = 4
plt.figure(ifig); plt.clf()
plt.plot(np.arange(np.shape(st_encoded)[0]), st_encoded)
plt.xlabel('observation')
plt.ylabel('value')
plt.title('Original: each curve represents the values for a particular feature')
plt.show()
    
ifig += 1
plt.figure(ifig); plt.clf()
plt.plot(np.arange(np.shape(st_encoded)[1]), st_encoded.T)
plt.xlabel('feature')
plt.ylabel('value')
plt.title('Original: each curve represents the values for a particular observation')
    
    # Plot the encoded data, just to see what it looks like
ifig += 1
plt.figure(ifig); plt.clf()
plt.plot(np.arange(np.shape(proj_down)[0]), proj_down)
plt.xlabel('observation')
plt.ylabel('value')
plt.title('Encoded: each curve represents the values for a particular feature')
    
ifig += 1
plt.figure(ifig); plt.clf()
plt.plot(np.arange(np.shape(proj_down)[1]), proj_down.T, '.-')
plt.xlabel('feature')
plt.ylabel('value')
plt.title('Encoded: each curve represents the values for a particular observation')
"""
    # Plot a particular test case at index i
i = 0
ifig += 1
plt.figure(ifig); plt.clf()
plt.subplot(211)
plt.plot(inputs[i],label = 'Original')
plt.plot(proj_down[i], label = 'Decoded')
plt.legend()
plt.subplot(212)
plt.plot(inputs[i] - proj_down[i], label = 'original - decoded')
plt.legend()
    
ifig += 1
plt.figure(ifig)
plt.title('Original - Decoded')
for i in range(np.shape(proj_down)[0]):
    plt.plot(proj_down[i] - proj_down[i])
""" 
    