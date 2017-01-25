# Script simple qui ajoute un rectangle a des images sur les visages grace aux landmark
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
import os
import select
import sys
import cv2
import dlib
import numpy as np

cascPath = '/home/pi/projets/librairies/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_alt_tree.xml'
eyePath = '/home/pi/projets/librairies/opencv-2.4.9/data/haarcascades/haarcascade_eye.xml'

PREDICTOR_PATH = "/home/pi/projets/librairies/dlib/shape_predictor_68_face_landmarks.dat"
cascade_path='/home/pi/projets/librairies/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_default.xml'

imgSave = '/home/pi/projets/face-reco/tests/dataset-save'

imgPath = '/home/pi/projets/face-reco/tests/dataset'
imgRectangle = '/home/pi/projets/face-reco/tests/dataset-rectangle'
imgCrop = '/home/pi/projets/face-reco/tests/crop'
imgCropAlign = '/home/pi/projets/face-reco/tests/crop-align'
imgDataset = '/home/pi/projets/face-reco/tests/dataset'
imgSave = '/home/pi/projets/face-reco/tests/dataset-save'

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


def get_landmarks(im):
    rects = cascade.detectMultiScale(im, 1.3,5)
    rectlist = []
    for (x, y, w, h) in rects:
    	#x,y,w,h =rects[0]
    	rect=dlib.rectangle(x,y,x+w,y+h)
    	a = np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
    	rectlist.append(a[0])
    return rectlist

def annotate_landmarks(im, landmarks):
    im = im.copy()
    droit = []
    gauche = []
    print "Landmark: {0}".format(landmarks)
    for l in landmarks:
	    if l is not None and len(l) >= 0:
		    for idx, point in enumerate(l):
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
			return im,droit,gauche

predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade = cv2.CascadeClassifier(cascade_path)
directory = read_images(imgDataset)
file = 0
for filename in directory:
    if filename != '.DS_Store' and filename !='.AppleDouble' and filename != '.Parent' and filename != 'cropfaces':
        fichier=filename;
        # Read the image
        image = cv2.imread(fichier,cv2.CV_LOAD_IMAGE_COLOR)
        land = annotate_landmarks(image,get_landmarks(image))
        if land is not None and len(land) >= 0:
        	annonate_img = land[0]
        	image_save = imgSave + "/img_" +str(file) + ".jpg"
        	print "{0}".format(land[1])
			#cv2.imwrite(image_save, annonate_img)
			#oeil gauche
			#cv2.rectangle(annonate_img,(land[1][0][0],land[1][1][1]),(land[1][3][0],land[1][4][1]),(0,255,0),2)
			#oeil droit
			#cv2.rectangle(annonate_img,(land[2][0][0],land[2][1][1]),(land[2][3][0],land[2][4][1]),(0,255,0),2)
			#On ajoute 20% dequart pour creer un rectangle autour des yeux
	        l_distance_x = ((land[1][3][0]-land[1][1][0])/5)
	        l_distance_y = ((land[1][4][1]-land[1][1][1])/5)
	        l_left_x = land[1][0][0] - l_distance_x
	        l_left_y = land[1][1][1] - l_distance_y
	        l_right_x = land[1][3][0] + l_distance_x
	        l_right_y = land[1][4][1] + l_distance_y
	        cv2.rectangle(annonate_img,(l_left_x,l_left_y),(l_right_x,l_right_y),(0,255,0),2)
	        #On passe a oeil droit
	        d_distance_x = ((land[2][3][0]-land[2][1][0])/5)
	        d_distance_y = ((land[2][4][1]-land[2][1][1])/5)
	        d_left_x = land[2][0][0] - d_distance_x
	        d_left_y = land[2][1][1] - d_distance_y
	        d_right_x = land[2][3][0] + d_distance_x
	        d_right_y = land[2][4][1] + d_distance_y
	        cv2.rectangle(annonate_img,(d_left_x,d_left_y),(d_right_x,d_right_y),(0,255,0),2)
	        cv2.imwrite(image_save, annonate_img)
	        print "{0} <--> {1} ".format(image_save, fichier)
	        cv2.imwrite(image_save, annonate_img)
	        file = file + 1