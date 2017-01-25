#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "/home/pi/projets/librairies/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.createLBPHFaceRecognizer()

def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in sorted(os.listdir(path)) if not f.endswith('.pgm')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    nam = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        print "{0}".format(image_path)
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split("_")[0])
        nom = os.path.split(image_path)[1].split("_")[1]
        images.append(image)
        print "face detected {0} - {1}".format(nbr,image_path)
        labels.append(nbr)
        nam.append((nom,nbr))
    # return the images list and labels list
    print "labels : {0}".format(labels)
    return images, labels,nam

# Path to the Yale Dataset
path = '/home/pi/projets/face-reco/tests/dataset'
# Call the get_images_and_labels function and get the face images and the 
# corresponding labels
images, labels,names = get_images_and_labels(path)
#cv2.destroyAllWindows()

# Perform the tranining
print "training"
recognizer.train(images, np.array(labels))

# Append the images with the extension .sad into image_paths
path2 = '/home/pi/projets/face-reco/tests/recotest'
print "listing"
image_paths = [os.path.join(path2, f) for f in sorted(os.listdir(path2)) if f.endswith('.JPG')]
print "debut reco"
for image_path in image_paths:
    print "--> Reco image sur {0}".format(image_path)
    predict_image_pil = Image.open(image_path).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)
    print "{0} visages trouves".format(len(faces))
    for (x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
        nbr_actual = int(os.path.split(image_path)[1].split("_")[1].split(".")[0])
        print "nbre predicted {0}".format(nbr_predicted)
        cv2.putText(predict_image, str(nbr_predicted), ((x,y)), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0, 0, 255))
        if nbr_actual == nbr_predicted:
            print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
        else:
            print "{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted)
    if len(faces) != 0:
        cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
        cv2.waitKey(1000)
