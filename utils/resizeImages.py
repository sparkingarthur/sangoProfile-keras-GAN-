#coding = utf-8
import cv2
import os
import PIL.Image as Image
import shutil
import re
from tqdm import tqdm

# img = Image.open('1.jpg')
# out = img.resize((50,60))
# out.save('2.jpg')

# img = cv2.imread('1.jpg')
# print(img.shape)
# height = 64
# width = 64
# newim = cv2.resize(img,(width,height),interpolation=cv2.INTER_AREA)
# cv2.imwrite("resized_test.jpg",newim,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
# cv2.imshow('image',newim)
# cv2.waitKey(0)
srcdirPath = '../mypic'
desdirPath = '../resizedpic'
files = os.listdir(srcdirPath)
objwidth = 64
objheight = 64
for img in tqdm(files):
    srcpath = os.path.join(srcdirPath,img)
    filename = re.findall(r'\d+',img)[1]
    despath = os.path.join(desdirPath,filename+'.jpg')
    oriimg = cv2.imread(srcpath)
    newim = cv2.resize(oriimg,(objwidth,objheight),interpolation=cv2.INTER_AREA)
    cv2.imwrite(despath,newim)
    #print(filename+" finished")

