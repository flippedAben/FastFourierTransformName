#!/usr/bin/env python
import numpy as np
import cmath
import matplotlib.pyplot as plt
from matplotlib import animation


fig, ax = plt.subplots()
# Size of axes
X = 500
Y = 500
X = 200
Y = 100

# Complex points of a figure
pts = []
# pts.append(1+0j)
# pts.append(1+1j)
# pts.append(0+1j)
# pts.append(-1+1j)
# pts.append(-1+0j)
# pts.append(-1-1j)
# pts.append(0-1j)
# pts.append(1-1j)

# M
pts.append(20+20j-50)
pts.append(20+25j-50)

pts.append(20+30j-50)
pts.append(20+35j-50)

pts.append(20+40j-50)
pts.append(25+35j-50)

pts.append(30+30j-50)
pts.append(35+35j-50)

pts.append(40+40j-50)
pts.append(40+35j-50)

pts.append(40+30j-50)
pts.append(40+20j-50)

# I
pts.append(50+20j-50)
pts.append(55+20j-50)

pts.append(60+20j-50)
pts.append(65+20j-50)

pts.append(70+20j-50)
pts.append(65+20j-50)

pts.append(60+20j-50)
pts.append(60+25j-50)

pts.append(60+30j-50)
pts.append(60+35j-50)

pts.append(60+40j-50)
pts.append(55+40j-50)

pts.append(50+40j-50)
pts.append(55+40j-50)

pts.append(60+40j-50)
pts.append(65+40j-50)
pts.append(70+40j-50)

# L
pts.append(80+40j-50)
pts.append(80+35j-50)
pts.append(80+30j-50)
pts.append(80+25j-50)
pts.append(80+20j-50)
pts.append(85+20j-50)
pts.append(90+20j-50)
pts.append(95+20j-50)
pts.append(100+20j-50)

# L
pts.append(80+40j-50+30)
pts.append(80+35j-50+30)
pts.append(80+30j-50+30)
pts.append(80+25j-50+30)
pts.append(80+20j-50+30)
pts.append(85+20j-50+30)
pts.append(90+20j-50+30)
pts.append(95+20j-50+30)
pts.append(100+20j-50+30)
# E
pts.append(140+20j-50)
pts.append(140+25j-50)
pts.append(140+30j-50)
pts.append(140+35j-50)
pts.append(140+40j-50)
pts.append(145+40j-50)
pts.append(150+40j-50)
pts.append(155+40j-50)

pts.append(160+40j-50)
pts.append(155+37.5j-50)
pts.append(150+35j-50)
pts.append(145+32.5j-50)
pts.append(140+30j-50)
pts.append(145+30j-50)
pts.append(150+30j-50)
pts.append(155+30j-50)

pts.append(160+30j-50)
pts.append(155+27.5j-50)
pts.append(150+25j-50)
pts.append(145+22.5j-50)

pts.append(140+20j-50)
pts.append(145+20j-50)
pts.append(150+20j-50)
pts.append(155+20j-50)

pts.append(160+20j-50)
# R
pts.append(170+20j-50)
pts.append(170+25j-50)
pts.append(170+30j-50)
pts.append(170+35j-50)

pts.append(170+40j-50)
pts.append(175+40j-50)
pts.append(180+40j-50)
pts.append(185+40j-50)
pts.append(190+40j-50)
pts.append(190+35j-50)
pts.append(190+30j-50)
pts.append(185+30j-50)
pts.append(180+30j-50)
pts.append(185+25j-50)
pts.append(190+20j-50)

# Calculate the Fourier Transform
fft_result = map(lambda x: x/len(pts), np.fft.fft(pts))

# fft result in polar form (radius, phase shift)
fft_polar = map(cmath.polar, fft_result)

# Number of circles for FFT
N = len(fft_polar)

# Make a list of potential circles that will be animated
# Circles are of the form: pots[i] = (radius, phase shift, angular frequency)
pots = []

# af for angular frequency
for af in range(1,N/2+1):
    # Only animate the circle if the radius is big enough
    if fft_polar[af][0] > -1000:
        pots.append((fft_polar[af][0], fft_polar[af][1], af))
    # if we are at i = n/2, skip adding the negative frequency
    if af == N - af:
        continue
    # Also look the negative frequencies
    if fft_polar[af][0] > -1000:
        pots.append((fft_polar[N-af][0], fft_polar[N-af][1], -1*af))

# Sort pots by radius from largest to smallest except for stationary circle
sorted(pots, key=lambda x: x[0],reverse=True)
pots.insert(0, (fft_polar[0][0], fft_polar[0][1], 0))

# List of circles
circles = []
for pot in pots:
    circles.append(plt.Circle( (0,0), pot[0], fill=False))

for circle in circles:
    ax.add_artist(circle)

# Now pots[i] should correspond to circles[i]

NC = len(circles)

# List of lines
lines = []
for i in range(1,NC):
    lines.append(plt.Line2D(circles[i-1].center, circles[i].center, lw=2))
lines.append(plt.Line2D(circles[NC-1].center, circles[NC-1].center, lw=2))

for i in range(NC):
    ax.add_line(lines[i])

# List of points left off of circle
marks = []

def init():
    # center of circle 0 = (0,0)
    # Find initial center of circle i
    for i in range(1,NC):
        x = circles[i-1].center[0] + circles[i-1].radius * np.cos(pots[i-1][1])
        y = circles[i-1].center[1] + circles[i-1].radius * np.sin(pots[i-1][1])
        circles[i].center = (x,y)

    # Find lines connecting center of circles
    for i in range(NC-1):
        lines[i].set_xdata([circles[i].center[0], circles[i+1].center[0]])
        lines[i].set_ydata([circles[i].center[1], circles[i+1].center[1]])

    # Find lines[NC-1]
    x = circles[NC-1].center[0] + circles[NC-1].radius * np.cos(pots[NC-1][1])
    y = circles[NC-1].center[1] + circles[NC-1].radius * np.sin(pots[NC-1][1])
    lines[NC-1].set_xdata([circles[NC-1].center[0], x])
    lines[NC-1].set_ydata([circles[NC-1].center[1], y])

    return tuple(circles) + tuple(lines) + tuple(marks)

def animate(degrees):
    t = np.radians(degrees)

    # Update centers of circles
    for j in range(1,NC):
        x = circles[j-1].center[0] + circles[j-1].radius * \
            np.cos(pots[j-1][1] + t*pots[j-1][2])
        y = circles[j-1].center[1] + circles[j-1].radius * \
            np.sin(pots[j-1][1] + t*pots[j-1][2])
        circles[j].center = (x,y)

    # Update endpoints of lines connecting center of circles
    for i in range(NC-1):
        lines[i].set_xdata([circles[i].center[0], circles[i+1].center[0]])
        lines[i].set_ydata([circles[i].center[1], circles[i+1].center[1]])

    # Update lines[NC-1]
    x = circles[NC-1].center[0] + circles[NC-1].radius * np.cos(pots[NC-1][1] +
            t*pots[NC-1][2])
    y = circles[NC-1].center[1] + circles[NC-1].radius * np.sin(pots[NC-1][1] +
            t*pots[NC-1][2])
    lines[NC-1].set_xdata([circles[NC-1].center[0], x])
    lines[NC-1].set_ydata([circles[NC-1].center[1], y])

    if degrees%2 == 0:
        marks.append(plt.Circle((x,y),1.5,fill=True,color='r'))
        ax.add_artist(marks[-1]) 
    if degrees == 359:
        del marks[:]
        for mark in marks:
            mark.set_radius(0)
    
    return tuple(circles) + tuple(lines) + tuple(marks)


plt.xlim(-X/2,X)
plt.ylim(-Y,Y)
anime = animation.FuncAnimation(fig, animate, init_func=init,
        frames=360,
        interval = 10, blit=True)

anime.save('miller.gif', dpi=80, writer='imagemagick')
plt.show()

