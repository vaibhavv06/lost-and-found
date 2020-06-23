import dlib
import imageio
import numpy as np
import os

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def get_face_encodings(image):
  image = imageio.imread(image)
  detected_faces = face_detector(image,1)
  shapes_faces = [shape_predictor(image,face) for face in detected_faces]
  return [np.array(model.compute_face_descriptor(image,face_pose,1)) for face_pose in shapes_faces]

def compare_face_encodings(face):
  return (np.linalg.norm(face_encodings - face,axis=1)<=TOLERANCE)

def find_match(face):
  matches = compare_face_encodings(face)
  count = 0
  for match in matches:
    if match:
      return int(names[count])
    count += 1
  return -2

TOLERANCE = 0.6
face_encodings = []
image_filenames = filter(lambda x: x.endswith('.jpg'),os.listdir('face_images/'))
image_filenames = sorted(image_filenames)
names = [x[:-4] for x in image_filenames]

def data_config():
  paths_to_images = ['face_images/' + x for x in image_filenames]

  for path_to_image in paths_to_images:
    face_encodings_in_image = get_face_encodings(path_to_image)
  
    if len(face_encodings_in_image) != 1:
      print("Please change image: " + path_to_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
      exit()
    
    face_encodings.append(face_encodings_in_image[0])
  
  print(len(face_encodings), "images in images folder")
  

# test_filenames = filter(lambda x: x.endswith('.jpg'),os.listdir('test/'))
# paths_to_test_images = ['test/' + x for x in test_filenames]

def test(path):
  face_encodings_in_image = get_face_encodings(path)
  
  if len(face_encodings_in_image) != 1:
    return -1
  match = find_match(face_encodings_in_image[0])
  return match

# def test():
#   print(paths_to_test_images)
#   for path_to_test_image in paths_to_test_images:
#     face_encodings_in_image = get_face_encodings(path_to_test_image)

#     if len(face_encodings_in_image) != 1:
#       print("Please change image: " + path_to_test_image + " - it has " + str(len(face_encodings_in_image)) + " faces; it can only have one")
#       exit()

#     match = find_match(face_encodings,names,face_encodings_in_image[0])
#     print(path_to_test_image,match)
