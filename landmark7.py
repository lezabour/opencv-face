# Ajoute les landmarks sur les images + sauvegarde un CROP + ALIGN en fonction des yeux
# si aucun visage detecte, on test un autre haarcascade pour etre sur
# plusieurs visages gesres
# FACE_POINTS = list(range(17, 68))
#        MOUTH_POINTS = list(range(48, 61))
#        RIGHT_BROW_POINTS = list(range(17, 22))
#        LEFT_BROW_POINTS = list(range(22, 27))
#        RIGHT_EYE_POINTS = list(range(36, 42))
#        LEFT_EYE_POINTS = list(range(42, 48))
#        NOSE_POINTS = list(range(27, 35))
#        JAW_POINTS = list(range(0, 17))
#        CHIN_POINTS=list(range(6,11))

import sys
# append facerec to module search path
sys.path.append("/home/pi/projets/facerec/py/apps/")
sys.path.append("/home/pi/projets/facerec/py/")

import cv2
from facedet.detector import SkinFaceDetector
import numpy as np
import os

import sys, math, Image
import os
import select
import sys
import cv2
import dlib
import numpy as np
import random

cascPath = '/home/pi/projets/librairies/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_alt2.xml'
eyePath = '/home/pi/projets/librairies/opencv-2.4.9/data/haarcascades/haarcascade_eye.xml'

PREDICTOR_PATH = "/home/pi/projets/librairies/dlib/shape_predictor_68_face_landmarks.dat"
cascade_path='/home/pi/projets/librairies/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_alt2.xml'

imgSave = '/home/pi/projets/face-reco/tests/dataset-save'

imgPath = '/home/pi/projets/face-reco/tests/dataset'
imgRectangle = '/home/pi/projets/face-reco/tests/dataset-rectangle'

imgLandmark = '/home/pi/projets/face-reco/tests/landmark'
imgCrop = '/home/pi/projets/face-reco/tests/crop'
imgCropAlign = '/home/pi/projets/face-reco/tests/crop-align'

imgDataset = '/home/pi/projets/face-reco/tests/dataset'

imgSave = '/home/pi/projets/face-reco/tests/dataset-save'

def Distance(p1,p2):
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  return math.sqrt(dx*dx+dy*dy)

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
  if (scale is None) and (center is None):
    return image.rotate(angle=angle, resample=Image.ANTIALIAS)
  nx,ny = x,y = center
  sx=sy=1.0
  if new_center:
    (nx,ny) = new_center
  if scale:
    (sx,sy) = (scale, scale)
  cosine = math.cos(angle)
  sine = math.sin(angle)
  a = cosine/sx
  b = sine/sx
  c = x-nx*a-ny*b
  d = -sine/sy
  e = cosine/sy
  f = y-nx*d-ny*e
  return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):
  # calculate offsets in original image
  offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
  offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
  # get the direction
  eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
  # calc rotation angle in radians
  rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))
  # distance between them
  dist = Distance(eye_left, eye_right)
  # calculate the reference eye-width
  reference = dest_sz[0] - 2.0*offset_h
  # scale factor
  scale = float(dist)/float(reference)
  # rotate original around the left eye
  image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
  # crop the rotated image
  crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
  crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
  image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
  # resize it
  image = image.resize(dest_sz, Image.ANTIALIAS)
  return image
  
  
def extract_faces(src_dir, dst_dir):
    """
    Extracts the faces from all images in a given src_dir and writes the extracted faces
    to dst_dir. Needs a facedet.Detector object to perform the actual detection.
    
    Args:
        src_dir [string] 
        dst_dir [string] 
        detector [facedet.Detector]
        face_sz [tuple] 
    """
    if not os.path.exists(dst_dir):
        try:
            os.mkdir(dst_dir)
        except:
            raise OSError("Can't create destination directory (%s)!" % (dst_dir))
    fichier = 0
    gauche = []
    droit = []
    landmark =[]
    for dirname, dirnames, filenames in os.walk(src_dir):
        for subdir in dirnames: 
                src_subdir = os.path.join(dirname, subdir)
                dst_subdir = os.path.join(dst_dir,subdir)
                dst_subdir_landmark = os.path.join(imgLandmark,subdir)
                dst_subdir_crop = os.path.join(imgCrop,subdir)
                dst_subdir_crop_align = os.path.join(imgCropAlign,subdir)
                if not os.path.exists(dst_subdir_landmark):
                    try:
                        os.mkdir(dst_subdir_landmark)
                    except:
                        raise OSError("Can't create destination directory (%s)!" % (dst_subdir_landmark))
                if not os.path.exists(dst_subdir_crop_align):
                    try:
                        os.mkdir(dst_subdir_crop_align)
                    except:
                        raise OSError("Can't create destination directory (%s)!" % (dst_subdir_crop_align))
                if not os.path.exists(dst_subdir_crop):
                    try:
                        os.mkdir(dst_subdir_crop)
                    except:
                        raise OSError("Can't create destination directory (%s)!" % (dst_subdir_crop))
                if not os.path.exists(dst_subdir):
                    try:
                        os.mkdir(dst_subdir)
                    except:
                        raise OSError("Can't create destination directory (%s)!" % (dst_dir))
                for filename in sorted(os.listdir(src_subdir)):
                    name, ext = os.path.splitext(filename)
                    src_fn = os.path.join(src_subdir,filename)
                    dst_subdir_landmark = os.path.join(os.path.join(imgLandmark,subdir),filename)
                    dst_subdir_crop_align = os.path.join(os.path.join(imgCropAlign,subdir),filename)
                    print "----------------------------------------"
                    print "> ouverture fichier {0} - {1}".format(fichier,src_fn)
                    im = cv2.imread(src_fn)
                    cv2.imshow('image',im)
                    cv2.waitKey(1000)
                    i = 0
                    #rects = detector.detect(im)
                    faces = faceCascade.detectMultiScale(im, scaleFactor=1.1, minNeighbors=5)
                    print "found {0} faces ".format(len(faces))
                    #if len(faces) == 0:
                    #    faces = faceCascade2.detectMultiScale(im, scaleFactor=1.1, minNeighbors=5)
                    for (x, y, w, h) in (faces):
                            gauche = []
                            droit = []
                            print "----> Face {0}".format(i)
                            sub_face = im[y:y+h, x:x+w]
                            dst_subdir_crop = os.path.join(os.path.join(imgCrop,subdir),str(i)+"_"+filename)
                            dst_subdir_crop_align = os.path.join(os.path.join(imgCropAlign,subdir),str(i)+"_"+filename)
                            i = i + 1
                            cv2.imwrite(dst_subdir_crop, sub_face)
                            print "Crop -> {0}".format(dst_subdir_crop)
                            print "CropAlign -> {0}".format(dst_subdir_crop_align)
                            rect = dlib.rectangle(x,y,x+w,y+h)
                            print "dlib rect {0} {1} {2} {3}".format(x,y,x+w,y+h)
                            landmark = np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
                            print "landmark "
                            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0),2)
                            print "rectangle "
                            for idx, point in enumerate(landmark):
                                    pos = (point[0, 0], point[0, 1])
                                    if idx in list(range(42, 48)):
                                        droit.append(pos)
                                            #print "Right Eye : {0} - {1}".format(idx,pos)
                                    if idx in list(range(36, 42)):
                                        gauche.append(pos)
                                            #print "Left Eye : {0} - {1}".format(idx,pos)
                                    cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
                                    cv2.circle(im, pos, 3, color=(0, 255, 255))
                            print ".........debut oeil "
                            cv2.imshow('image',im)
                            cv2.waitKey(1000)
                            l_distance_x = (gauche[3][0]-gauche[0][0])/5
                            l_distance_y = ((gauche[4][1]-gauche[1][1])/5)
                            l_left_x = gauche[0][0] - l_distance_x
                            l_left_y = gauche[1][1] - l_distance_y
                            l_right_x = gauche[3][0] + l_distance_x
                            l_right_y = gauche[4][1] + l_distance_y
                            d_distance_x = ((droit[3][0]-droit[0][0])/5)
                            d_distance_y = ((droit[4][1]-droit[1][1])/5)
                            d_left_x = droit[0][0] - d_distance_x
                            d_left_y = droit[1][1] - d_distance_y
                            d_right_x = droit[3][0] + d_distance_x
                            d_right_y = droit[4][1] + d_distance_y
                            left_eye_center = (((l_left_x +l_right_x)/2) ,((l_left_y + l_right_y)/2))
                            right_eye_center = (((d_left_x +d_right_x)/2) ,((d_left_y + d_right_y)/2))
                            print "Left Eye : {0} - centre {1}".format(gauche,left_eye_center) 
                            print "Right Eye : {0} - centre {1}".format(droit,right_eye_center)
                            print "fin calcul on ajout les rectangle oeil "
                            cv2.rectangle(im,(l_left_x,l_left_y),(l_right_x,l_right_y),(0,255,0),2)
                            cv2.rectangle(im,(d_left_x,d_left_y),(d_right_x,d_right_y),(0,255,0),2)
                            print "fin oeil - debut crop face"
                            image =  Image.open(src_fn)
                            CropFace(image, eye_left= left_eye_center, eye_right= right_eye_center, offset_pct=(0.25,0.25), dest_sz=(100,100)).save(dst_subdir_crop_align, quality=90)
                            print "..............fin crop face - face ou image suivante"
                    if len(faces)>0:
                        cv2.imwrite(dst_subdir_landmark, im)    
                    print "fin {0} fichier {1}".format(fichier,src_fn)
                    fichier = fichier +1
                    #cv2.destroyAllWindows()
                #cv2.imwrite(dst_subdir_landmark, im)                         

src_dir = imgDataset
dst_dir = imgCropAlign
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade = cv2.CascadeClassifier(cascade_path)
#directory = read_images(imgDataset)
file = 0
faceCascade = cv2.CascadeClassifier(cascPath)
#detector = SkinFaceDetector(threshold=0.3, cascade_fn="/home/pi/projets/librairies/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_alt2.xml")
cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
extract_faces(src_dir=src_dir, dst_dir=dst_dir)
