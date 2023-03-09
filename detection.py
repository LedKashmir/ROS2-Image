#!/usr/bin/env python3
import cv2
import numpy as np
import os
import itertools
from numpy.polynomial import polynomial as P
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

def cv_show(name, img):         #画图函数
    img=cv2.resize(img,(0,0),fx=0.5,fy=0.5)
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def generate_mask(image):
    lower = np.uint8([175, 175, 200])           #得到mask
    upper = np.uint8([255, 255, 255])           # lower_red和高于upper_red的部分分别变成0，lower_red～upper_red之间的值变成255,相当于过滤背景
    white_mask = cv2.inRange(image, lower, upper)
    return white_mask
 
def canny(image,white_mask):
    white_yellow_image = cv2.bitwise_and(image, image, mask=white_mask)  # 与操作去除背景
    filter = cv2.GaussianBlur(white_yellow_image, (5, 5), 0)
    gray_image = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY) # 转灰度图
    #cv_show("gray",gray_image)
    #边缘检测
    canny_img = cv2.Canny(gray_image, 35, 140,5)
    #cv_show("canny",canny_img)
    return canny_img

def LinesDetection(edge_image,image):
    roi_image=edge_image.copy()
    lines = cv2.HoughLinesP(roi_image, rho=1, theta=np.pi/180, threshold=50, minLineLength=150, maxLineGap=50)
    line_image = np.copy(image)
    if lines is None:
        return image,[]
    else:
        line_image=draw_lines(lines,line_image)
        return line_image, lines

def Longest(cluster):
    cluster=np.array(cluster)
    dist_list=[]
    for line in cluster:
        x1,y1,x2,y2 = line[0]
        dist=np.linalg.norm([x1-x2,y1-y2])
        dist_list.append(dist)
    index=np.argmax(dist_list,axis=0)
    longest_line=cluster[index]
    return longest_line

def merge_lines(lines):#线段聚类
    clusters = []
    idx = []
    for i,line in enumerate(lines):
        x1,y1,x2,y2 = line[0]
        if [x1,y1,x2,y2] in idx:
             continue
        parameters = P.polyfit([x1, x2],[y1, y2], 1)
        slope = parameters[1]#(y2-y1)/(x2-x1+0.001)
        intercept = parameters[0]#((y2+y1) - slope *(x2+x1))/2
        k = slope
        b = intercept
        d = np.sqrt(k**2+1)
        cluster = [line]
        for d_line in lines[i+1:]:
            x,y,xo,yo= d_line[0]
            mid_x = (x+xo)/2
            mid_y = (y+yo)/2
            distance = np.abs(k*mid_x-mid_y+b)/d
            if distance < 20  :
                cluster.append(d_line)
                idx.append(d_line[0].tolist())
            else:
                continue
        clusters.append(np.array(cluster))
    merged_lines = [Longest(cluster) for cluster in clusters]
    merged_lines = np.array(merged_lines)
    return merged_lines

def draw_lines(lines,line_image):
    for dline in lines:
        x1, y1, x2, y2 = dline[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0,255,0), 2, cv2.LINE_AA)
        cv2.circle(line_image, (x1, y1), 3, (255, 0, 0), 3)
        cv2.circle(line_image, (x2, y2), 3, (255, 0, 0), 3)
    return line_image

def draw_points(points,image):
    for point in points:
        point=point.astype(int)
        cv2.circle(image,(point[0],point[1]),3,(0,255,255),3)
    return image,points

def extract_vertice(lines):
    points=[]
    for i in range(lines.shape[0]):
        point1=lines[i][0][0:2]
        point2=lines[i][0][2:4]
        points.append(point1)
        points.append(point2)
    points=np.array(points)
    return points

def draw_poly(parking_spots,image):
    zeros = np.zeros((image.shape), dtype=np.uint8)
    mask = np.zeros((image.shape), dtype=np.uint8)
    for parking_spot in parking_spots:
        Points = np.array(parking_spot.exterior.coords, np.int32)
        mask=cv2.fillPoly(zeros, [Points], (0, 100, 100))
    image=cv2.addWeighted(image,1,mask,0.5,0)
    return image

def find_parking_spot(points):
    groups=np.array(list(itertools.combinations(np.arange(points.shape[0]), 4)))
    numSpots = 0
    spotArea = 38000  # Expected parking spot area
    parkingSpots = []
    centroid_points=[]
    for i in range(groups.shape[0]):#groups.shape[0]
        corner_points=points[groups[i,:],:]
        #print("corn",corner_points)
        centerPoint = np.mean(corner_points, axis=0)
        #print("cenc",centerPoint)
        distances = np.linalg.norm(corner_points - centerPoint, axis=1)
        #print("dis",distances)
        hasCollinearPoints = len(np.unique(corner_points[:,0]))==1 or len(np.unique(corner_points[:,1]))==1
        #print(hasCollinearPoints)
        if np.max(distances) - np.min(distances) < 50 and not hasCollinearPoints:
            # Compute the area of the rectangle
            pgon = Polygon(corner_points)
            hull = pgon.convex_hull
            if hull.is_valid and len(hull.exterior.coords)==5:
                if abs(hull.area - spotArea) < 8000:
                    # Check if the spot is occupied by a parking vehicle
                    centroid_point=[hull.centroid.x,hull.centroid.y]
                    centroid_points.append(centroid_point)
                    parkingSpots.append(hull)
                    numSpots += 1
    centroid_points=np.array(centroid_points)
    return parkingSpots,centroid_points

def operation(image):
    mask=generate_mask(image)
    edge_image=canny(image,mask)
    line_image,lines=LinesDetection(edge_image,image)
    cluster_lines=merge_lines(lines)
    cluster_img=draw_lines(cluster_lines,image)
    points=extract_vertice(cluster_lines)
    parking_spots, cent_points = find_parking_spot(points)
    image, center = draw_points(cent_points, cluster_img)
    image = draw_poly(parking_spots, image)
    return image
