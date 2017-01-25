# Ajoute les landmarks sur les images + sauvegarde un CROP + ALIGN en fonction des yeux
#pour avoir des images de dataset pour reco faciale
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
import sys, math, Image
import os
import select
import sys
import cv2
import dlib
import numpy as np
import random

cascPath = '/home/pi/projets/librairies/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_default.xml'
eyePath = '/home/pi/projets/librairies/opencv-2.4.9/data/haarcascades/haarcascade_eye.xml'

PREDICTOR_PATH = "/home/pi/projets/librairies/dlib/shape_predictor_68_face_landmarks.dat"
cascade_path='/home/pi/projets/librairies/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_default.xml'

imgSave = '/home/pi/projets/face-reco/tests/dataset-save'

imgPath = '/home/pi/projets/face-reco/tests/dataset'
imgRectangle = '/home/pi/projets/face-reco/tests/dataset-rectangle'
imgCrop = '/home/pi/projets/face-reco/tests/crop'
imgCropAlign = '/home/pi/projets/face-reco/tests/crop-align/'
imgDataset = '/home/pi/projets/face-reco/tests/dataset'
imgSave = '/home/pi/projets/face-reco/tests/dataset-save'

def Distance(p1,p2):
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  return math.sqrt(dx*dx+dy*dy)

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
  if (scale is None) and (center is None):
    return image.rotate(angle=angle, resample=resample)
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
  
def is_letter_input(letter):
	# Utility function to check if a specific character is available on stdin.
	# Comparison is case insensitive.
	if select.select([sys.stdin,],[],[],0.0)[0]:
		input_char = sys.stdin.read(1)
		return input_char.lower() == letter.lower()
	return False

def read_images(path, image_size=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A list [X, y, folder_names]

            X: The images, which is a Python list of np arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
            folder_names: The names of the folder, so you can display it in a prediction.
    """
    c = 0
    X = []
    y = []
    folder_names = []
    for dirname, dirnames, filenames in os.walk(path):
        #print dirname
        for subdirname in dirnames:
            folder_names.append(subdirname)
            #print subdirname
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                if filename != '.DS_Store' and filename !='.AppleDouble' and filename != '.Parent' and filename != 'cropfaces':
                    try:
                        #print "--->{0}/{1}/{2}".format(dirname,subdirname,filename)
                        #print "## {0}".format(os.path.join(subject_path, filename))
                        filefinal = os.path.join(subject_path, filename)
                        #print filefinal
                        #im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                        if(os.path.isfile(filefinal)):
                        	y.append(filefinal)
                    except IOError, (errno, strerror):
                        print "I/O error({0}): {1}".format(errno, strerror)
                    except:
                        print "Unexpected error:", sys.exc_info()[0]
                        raise
    return y
#g [(261, 606), (295, 586), (338, 585), (377, 609), (337, 619), (294, 622)]#
#		42			43			44		45				46		47#
#	0.0, 0.1	1.0, 1.1	2.0, 2.1	3.0, 3.1	4.0, 4.1	5.0, 5.1
#d [(522, 615), (559, 596), (598, 600), (626, 620), (597, 632), (559, 628)]

def def_landmark(im,fichier):
    faces = faceCascade.detectMultiScale(
        im,
        scaleFactor=1.3,
        minNeighbors=5,
    )
    print "Found {0} faces!".format(len(faces))
    rd = random.randint(0,500000);
    cropalign = imgCropAlign+"arnie_"+ str(rd) + ".jpg"
    image_save = imgSave + "/img_" + str(rd) + ".jpg"
    count = 1
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
        rect=dlib.rectangle(x,y,x+w,y+h)
        landmark = np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
        gauche = []
        droit = []
        for idx, point in enumerate(landmark):
            pos = (point[0, 0], point[0, 1])
            if idx in list(range(42, 48)):
                droit.append(pos)
                print "Right Eye : {0} - {1}".format(idx,pos)
            if idx in list(range(36, 42)):
                gauche.append(pos)
                print "Left Eye : {0} - {1}".format(idx,pos)
            cv2.putText(im, str(idx), pos,
	                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
	                    fontScale=0.4,
	                    color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
        l_distance_x = (gauche[3][0]-gauche[0][0])/5
        l_distance_y = ((gauche[4][1]-gauche[1][1])/5)
        l_left_x = gauche[0][0] - l_distance_x
        l_left_y = gauche[1][1] - l_distance_y
        l_right_x = gauche[3][0] + l_distance_x
        l_right_y = gauche[4][1] + l_distance_y
        cv2.rectangle(im,(l_left_x,l_left_y),(l_right_x,l_right_y),(0,255,0),2)
        d_distance_x = ((droit[3][0]-droit[0][0])/5)
        d_distance_y = ((droit[4][1]-droit[1][1])/5)
        d_left_x = droit[0][0] - d_distance_x
        d_left_y = droit[1][1] - d_distance_y
        d_right_x = droit[3][0] + d_distance_x
        d_right_y = droit[4][1] + d_distance_y
        cv2.rectangle(im,(d_left_x,d_left_y),(d_right_x,d_right_y),(0,255,0),2)
        #rd = random.randint(0,500000);
        image =  Image.open(fichier)
        #cropalign = imgCropAlign+"arnie_"+ str(rd) + ".jpg"
        CropFace(image,
         eye_left= (((l_left_x +l_right_x)/2) ,((l_left_y + l_right_y)/2)),
         eye_right= (((d_left_x +d_right_x)/2) ,((d_left_y + d_right_y)/2 )),
         offset_pct=(0.25,0.25),
         dest_sz=(100,100)).save(cropalign)
	#image_save = imgSave + "/img_" + str(rd) + ".jpg"
    cv2.imwrite(image_save, im)
    return im


predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade = cv2.CascadeClassifier(cascade_path)
directory = read_images(imgDataset)
file = 0
faceCascade = cv2.CascadeClassifier(cascPath)
for filename in directory:
    if filename != '.DS_Store' and filename !='.AppleDouble' and filename != '.Parent' and filename != 'cropfaces':
        fichier=filename;
        # Read the image
        image = cv2.imread(fichier,cv2.CV_LOAD_IMAGE_COLOR)
        land = def_landmark(image,fichier)
        image_save = imgSave + "/img_" +str(file) + ".jpg"
        cv2.imwrite(image_save, land)
        print "{0}".format(image_save)
        file = file + 1
