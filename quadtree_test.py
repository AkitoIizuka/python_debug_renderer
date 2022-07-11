from cmath import inf
from ctypes import sizeof
from ctypes.wintypes import WORD
from platform import node
from random import sample
from smtplib import SMTPHeloError
from tkinter import Y
from xml.etree.ElementTree import PI
from xmlrpc.client import Boolean
from matplotlib.pyplot import specgram
import numpy as np
from numpy import random, source
from math import sqrt, cos, sin, atan2, floor
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

# Define header function
M_PI = 3.1415926535


def cosineSampleHemisphere(u):
    r = sqrt(u[0])
    theta = 2 * M_PI * u[1]
    x = r * cos(theta)
    y = r * sin(theta)
    return np.array([x, y, sqrt(max(0.0, 1 - u[0]))])


def cross(a, b):
    return np.array([a[1]*b[2]-b[1]*a[2],
                     a[2]*b[0]-b[2]*a[0],
                     a[0]*b[1]-b[0]*a[1]])


def DistanceToLine(origin, dir, point):
    return np.linalg.norm(cross(dir, point - origin)) / np.linalg.norm(dir)


def clamp(a, low, high):
    if a < low:
        return low
    if a > high:
        return high
    return a


def World2Spherical(dir):
    cosTheta = clamp(dir[1], -1.0, 1.0)
    y = atan2(dir[2], dir[0])
    if y < 0:
        y = y + 6.28318530717958
    # xy = np.array([(cosTheta + 1)*.5, y * .5 * 0.318309886183790]); #
    xy = np.array([cosTheta, y])
    return xy

def norm(dir):
    a = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
    dir = dir / a
    return dir

def Spherical2World(xy):
    dir = np.zeros([3], dtype=float)
    dir[1] = xy[0]
    dir[0] = cos(xy[1])
    dir[2] = sin(xy[1])
    return norm(dir)

light = np.zeros((2, 3), dtype=np.float32)

# Parameter Definision
sampleNum = 1
pos = np.array([0, 0, 0])
light[0] = np.array([10, 10, 0]) # Green
light[1] = np.array([-10, 10, 0]) # Red
radius = 1.5  # of light
defCW = True
depth = 16
threshold = 0.001
n = np.array([0.8, (2/3)*np.pi])

# quadtree
class GuideQuadNode:
    isLeaf = False
    intensity = 0.
    mid = np.array([0., np.pi])

    def __init__(self, isLeaf, intensity, mid):
        self.isLeaf = isLeaf
        self.intensity = intensity
        self.mid = mid


def buildQuadTree(tree: GuideQuadNode, depth, threshold):
    index = 0
    h = 0
    while(tree[index].intensity >= tree[0].intensity * threshold):
        h += 1
        if(h > depth):
            break
        dTheta = 1 / pow(2, h)
        dPhi =  np.pi / pow(2, h)
        index = int((1/3)*(pow(4, h) - 1))
        j = 0
        for i in range(pow(4, h)):
            j = index + i
            tree[floor((j - 1) / 4)].isLeaf = False
            m = np.array([tree[floor((j - 1) / 4)].mid[0], tree[floor((j - 1) / 4)].mid[1]])
            tmp = (j - 1) % 4
            if(tmp % 2):
                m[0] = m[0] + dTheta
            else:  
                m[0] = m[0] - dTheta
            if(floor(tmp / 2)):
                m[1] = m[1] + dPhi
            else:
                m[1] = m[1] - dPhi
            tree.append(GuideQuadNode(True, tree[floor((j - 1) / 4)].intensity / 4, m))

j = 0
samples = np.zeros((sampleNum, 3), dtype=np.float32)

tree = []

hit = False
while(not hit):
    p = np.array([0, 0, 0])
    dir = cosineSampleHemisphere(random.uniform(size=2))
    pdf = dir[2] / M_PI
    dir = np.array([dir[0], dir[2], dir[1]])
    dist = np.zeros((2), dtype=np.float32)
    dist[0] = DistanceToLine(p, dir, light[0])
    dist[1] = DistanceToLine(p, dir, light[1])
    # hit = (distance <= radius)
    # FIX: for multiple light source
    if dist[0] <= radius:
        distance = dist[0]
        hit = True
    elif dist[1] <= radius:
        distance = dist[1]
        hit = True
    else:
        continue
    f = 1
    val = f/pdf
    tree.append(GuideQuadNode(True, val, np.array([0., np.pi])))

buildQuadTree(tree, depth, threshold)
i = 0
height = 0
h = len(tree)

while(i < 10):
    ly = int((1/3)*(pow(4, i) - 1))
    uy = int((1/3)*(pow(4, i + 1) - 1))
    if (ly <= h and h < uy):
        height = i - 1
        break
    i += 1

for i in range(int((1/3)*(pow(4, height) - 1)), int((1/3)*(pow(4, height + 1) - 1))):
    hit = False
    p = np.array([0, 0, 0])
    dev = pow(2, height)
    dTheta = 1 / dev
    dPhi = np.pi / dev
    lowTheta = tree[i].mid[0] - dTheta
    upTheta = tree[i].mid[0] + dTheta
    lowPhi = tree[i].mid[1] - dPhi
    upPhi = tree[i].mid[1] + dPhi
    # print(lowTheta, " ", lowPhi)
    count = 0
    while(not hit):
        tmpPos = np.array([random.uniform(lowTheta, upTheta,
                          size=1), random.uniform(lowPhi, upPhi, size=1)])
        dir = Spherical2World(tmpPos)
        pdf = dir[1] / M_PI
        dist = np.zeros((2), dtype=np.float32)
        dist[0] = DistanceToLine(p, dir, light[0])
        dist[1] = DistanceToLine(p, dir, light[1])
        # hit = (distance <= radius)
        count += 1
        if(count > 100):
            f = 1
            val = clamp(f/pdf, 0, inf)
            val = f/pdf
            tree[i].intensity = 0  # val
            # print("non val is ", tree[i].intensity)
            break
        if dist[0] <= radius:
            distance = dist[0]
            hit = True
        elif dist[1] <= radius:
            distance = dist[1]
            hit = True
        else:
            continue
        f = 1
        val = f/pdf
        val = clamp(val, 0, inf)
        tree[i].intensity = val
        # print("val is ", val)

j = height - 1
while(j >= 0):
    for i in range(int((1/3)*(pow(4, j) - 1)), int((1/3)*(pow(4, j + 1) - 1))):
        tree[i]. intensity = tree[4*i + 1].intensity + \
            tree[4*i + 2].intensity + tree[4*i + 3].intensity + tree[4*i + 4].intensity
    j -= 1

def sampleGuidingPath(tree, n):
    index = 0
    height = 0
    while(not tree[index].isLeaf):
        height += 1
        totalWeight = tree[index].intensity
        targetWeight = totalWeight * random.uniform(0, 1)
        for i in range(1, 5):
            # print(4*index+i, tree[4*index+i].intensity) # com out
            pass
        i = 1
        while(i < 4):
            if(targetWeight < tree[4*index + i].intensity):
                break
            targetWeight -= tree[4*index + i].intensity
            i += 1
        index = 4 * index + i
    dev = pow(2, height)
    dTheta = 1 / dev
    dPhi = np.pi / dev
    lowTheta = tree[index].mid[0] - dTheta
    upTheta = tree[index].mid[0] + dTheta
    lowPhi = tree[index].mid[1] - dPhi
    upPhi = tree[index].mid[1] + dPhi
    tmpPos = np.array([random.uniform(lowTheta, upTheta,
                          size=1), random.uniform(lowPhi, upPhi, size=1)])
    w = Spherical2World(tmpPos)
    # print(index) # com out
    return w

hitCounter = np.array([0, 0], dtype=int)

# print(sampleGuidingPath(tree, n))
# print(sampleGuidingPathWithCosineWeight(tree, n))

# hitCounter = np.array([0, 0], dtype=int)
# for i in range(sampleNum):
#     wo = sampleGuidingPath(tree, n)
#     wcw = sampleGuidingPathWithCosineWeight(tree, n)
#     dist = np.zeros((4), dtype=np.float32)
#     dist[0] = DistanceToLine(p, wo, light[0])
#     dist[1] = DistanceToLine(p, wo, light[1])
#     dist[2] = DistanceToLine(p, wcw, light[1])
#     dist[3] = DistanceToLine(p, wcw, light[1])
#     # hit = (distance <= radius)
#     distance = 0
#     # print(dist)
#     # print("wo: ", wo)
#     # print("wc: ", wcw)
#     # print(" ")
#     if dist[0] <= radius:
#         distance = dist[0]
#         hitCounter[0] += 1
#     elif dist[1] <= radius:
#         distance = dist[1]
#         hitCounter[0] += 1
#     if dist[2] <= radius:
#         distance = dist[2]
#         hitCounter[1] += 1
#     elif dist[3] <= radius:
#         distance = dist[3]
#         hitCounter[1] += 1
#     print(distance)

# print("Without Cosine Weight", hitCounter[0])
# print("With Cosine Weight", hitCounter[1])

pixels = np.zeros((256, 256, 3))
cpixels = np.zeros((256, 256, 3))
upixels = np.zeros((256, 256, 3))

x = np.zeros([256*256])
y = np.zeros([256*256])
z = np.zeros([256*256])

def EvaluateHeat(tree):
    for i in range(256):
        for j in range(256):
            wo = sampleGuidingPath(tree, n)
            dist = np.zeros((2), dtype=np.float32)
            p = np.array([-4 + 0.03125 * i,0,  -4 + 0.03125 * j])
            dist[0] = DistanceToLine(p, wo, light[0])
            dist[1] = DistanceToLine(p, wo, light[1])
            distance = 0
            # print(dist)
            # print("wo: ", wo)
            # print("wc: ", wcw)
            # print(" ")
            dir = np.zeros((3))
            isRed = False
            if dist[0] <= radius:
                dir = wo
                isRed = False
                distance = dist[0]
            elif dist[1] <= radius:
                dir = wo
                isRed = True
                distance = dist[1]
            if (distance != 0):
                pdf = dir[2] / M_PI
                # val = 1 / pdf
                val = 256
            else:
                val = 0
            if(isRed):
                pixels[i][j][2] = val
            else:
                pixels[i][j][1] = val

                
def EvaluateHeatUniform(snum):
        for i in range(256):
            for j in range(256):
                for k in range(snum):                
                    p = np.array([-4 + 0.03125 * i,0, -4 + 0.03125 * j])
                    dir = cosineSampleHemisphere(random.uniform(size=2))
                    dir = np.array([dir[0], dir[2], dir[1]])
                    dist = np.zeros((2))
                    dist[0] = DistanceToLine(p, dir,light[0])
                    dist[1] = DistanceToLine(p, dir,light[1])
                    distance = 0
                    isRed = False
                    if dist[0] <= radius:
                        isRed = False
                        distance = dist[0]
                    elif dist[1] <= radius:
                        isRed = True
                        distance = dist[1]
                    if (distance != 0):
                        val = 256
                    else:
                        val = 0
                    if(isRed):
                        upixels[i][j][2] = val
                    else:
                        upixels[i][j][1] = val
                    if(distance != 0):
                        break


print("start evaluation")
EvaluateHeat(tree)
cv2.imwrite('de.png', pixels)
print("export de.png")

print("start evaluation by uniform sampling")
EvaluateHeatUniform(1)
cv2.imwrite('uni.png', upixels)
print("export uni.png")
  
# # figureを生成する
# fig = plt.figure()
 
# # axをfigureに設定する
# ax = Axes3D(fig)
# # ax = fig.add_subplot(111)
 
# # axesに散布図を設定する
# ax.scatter(x, y, z, c='b')
# ax.set_xlim(-1.2, 1.2)
# ax.set_ylim(-1.2, 1.2)
# ax.set_zlim(-0.1, 1.2)
 
# # 表示する
# plt.show()