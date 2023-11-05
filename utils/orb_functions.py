import cv2
import numpy as np
import os

orb = cv2.ORB_create(nfeatures=1000)


def findDescription(data):
  desList = []
  for img in data:
    kp, des = orb.detectAndCompute(img, None)
    desList.append(des)
  return desList

def findRelevantKeypoints(img, desList):
  kp2, des2 = orb.detectAndCompute(img, None)
  bf = cv2.BFMatcher()
  matchList = []
  try:
    for des in desList:
      matches = bf.knnMatch(des, des2, k=2)
      good = []
      for m,n in matches:
        if m.distance < 0.75*n.distance:
          good.append([m])
      matchList.append(len(good))
  except:
    pass
  print(matchList)
  return matchList

# Devuleve el id de la imagen con mayor similitud y su porcentaje
def findMax(matchList, thres=13):
  maxValue = max(matchList)
  finalVal = -1
  accuracy = 0

  if len(matchList) != 0:
    finalVal = matchList.index(maxValue)
    if maxValue > thres:
      accuracy = 100.0
    else:
      matchList.pop(finalVal)
      accuracy = (maxValue/thres) - ((sum(matchList)/4)/100)

  return finalVal, accuracy