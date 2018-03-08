
import os  
from PIL import Image  
import xml.dom.minidom  
import numpy as np 

ImgPath = 'data/VOC2012/JPEGImages/'   
AnnoPath = 'data/VOC2012/Annotations/'  
savepath ="./data/train/"
image_pre="image"

i = 1  

def parse(html):
    xmlfile = AnnoPath + html  
  
    DomTree = xml.dom.minidom.parse(xmlfile)  
    annotation = DomTree.documentElement  

    filenamelist = annotation.getElementsByTagName('filename') #[<DOM Element: filename at 0x381f788>]  
    filename = filenamelist[0].childNodes[0].data  
    imgfile = ImgPath + filename
    objectlist = annotation.getElementsByTagName('object')  	

    size = annotation.getElementsByTagName('size')[0]
    width = size.getElementsByTagName('width')[0].childNodes[0].data
    height = size.getElementsByTagName('height')[0].childNodes[0].data
    depth = size.getElementsByTagName('depth')[0].childNodes[0].data

    img_list=[]

    for objects in objectlist:  
        namelist = objects.getElementsByTagName('name')  
        objectname = namelist[0].childNodes[0].data  
  
        bndbox = objects.getElementsByTagName('bndbox')  
        cropboxes = []  
        for box in bndbox:  
            try:  
                x1_list = box.getElementsByTagName('xmin')  
                x1 = int(x1_list[0].childNodes[0].data)  
                y1_list = box.getElementsByTagName('ymin')  
                y1 = int(y1_list[0].childNodes[0].data)  
                x2_list = box.getElementsByTagName('xmax')  
                x2 = int(x2_list[0].childNodes[0].data)  
                y2_list = box.getElementsByTagName('ymax')  
                y2 = int(y2_list[0].childNodes[0].data)  
                w = x2 - x1  
                h = y2 - y1  
  
                #global i
                img = Image.open(imgfile)  
                cropbox = (x1,y1,x2,y2)  
                cropedimg = img.crop(cropbox)  
                img_list.append(cropedimg)
                #cropedimg.save(savepath + '/' + objectname + '_' + str(i) + '.jpg')  
                #i += 1  
                '''
                width,height = img.size  
      
		    obj = np.array([x1,y1,x2,y2])  
		    shift = np.array([[0.8,0.8,1.2,1.2],[0.9,0.9,1.1,1.1],[1,1,1,1],[0.8,0.8,1,1],[1,1,1.2,1.2],\
			[0.8,1,1,1.2],[1,0.8,1.2,1],[(x1+w*1/6)/x1,(y1+h*1/6)/y1,(x2+w*1/6)/x2,(y2+h*1/6)/y2],\
			[(x1-w*1/6)/x1,(y1-h*1/6)/y1,(x2-w*1/6)/x2,(y2-h*1/6)/y2]])  
      
		    XYmatrix = np.tile(obj,(9,1))    
		    cropboxes = XYmatrix * shift  
	  
		    for cropbox in cropboxes:  
			# print 'cropbox:',cropbox  
			minX = max(0,cropbox[0])  
			minY = max(0,cropbox[1])  
			maxX = min(cropbox[2],width)  
			maxY = min(cropbox[3],height)  
      
			cropbox = (minX,minY,maxX,maxY)  
			cropedimg = img.crop(cropbox)  
			cropedimg.save(savepath + '/' + image_pre + '_' + str(i) + '.jpg')  
			i += 1  
                    '''
      
            except Exception, e:  
                print e 

    return img_list

def get_voc_date():
    res_list=[]
    for s in os.listdir(AnnoPath):
        res_list+=parse(s)
    print len(res_list)
    return res_list
