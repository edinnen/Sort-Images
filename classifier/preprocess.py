import cv2
import os
import re
import sys
import Augmentor

def augmentImages(folders):
    for folder in folders:
        folder_name='rawdata/' + folder
        p = Augmentor.Pipeline(source_directory=folder_name,save_format="jpg")
        p.flip_left_right(0.5)
        p.gaussian_distortion(probability=0.4, grid_width=7, grid_height=6, magnitude=6, corner="ul", method="in", mex=0.5, mey=0.5, sdx=0.05, sdy=0.05)

        p.rotate(0.3, 10,10)
        p.skew(0.4,0.5)
        p.skew_tilt(0.6,0.8)
        p.skew_left_right(0.5, magnitude=0.8)
        p.sample(10000)

def image_processing(raw_data,data_path,height,width):
    class_labels=[]
    category_count=0
    for i in os.walk(raw_data):
        if len(i[2])>0:
            if i[0] == 'augmented' or i[0] == 'data/augmented':
                continue
            counter=0
            images=i[2]
            # class_name=i[0].strip('/')
            class_name = re.sub('augmented/', '', i[0])
            class_labels.append(class_name)
            print("Resizing ", class_name)
            path=os.path.join(data_path,class_labels[category_count])
            for image in images:
                if image == '.DS_Store':
                    continue
                try:
                    im=cv2.imread('augmented/'+class_name+'/'+image)
                    im=cv2.resize(im,(height,width))
                except:
                    continue
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(os.path.join(path,str(counter)+'.jpg'),im)
                counter+=1
            category_count+=1
        else:
            number_of_classes=len(i[1])
            print(number_of_classes,i[1])
            class_labels=i[1][:]

if __name__=='__main__':
    height = 1024
    width = 768
    raw_data = 'rawdata'
    data_path = 'data'
    augment_folder = 'augmented'
    if not os.path.exists(augment_folder):
        augmentImages(next(os.walk(raw_data))[1])
    if not os.path.exists(data_path):
        image_processing(augment_folder, data_path, height, width)
