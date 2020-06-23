import cv2
import os
import sys
import numpy
import matplotlib.pyplot as plt
from enhance import image_enhance
from skimage.morphology import skeletonize, thin

SCORE_THRESHOLD = 25

def removedot(invertThin): 
  temp0 = numpy.array(invertThin[:])
  temp0 = numpy.array(temp0)
  temp1 = temp0/255
  temp2 = numpy.array(temp1)
  temp3 = numpy.array(temp2)

  enhanced_img = numpy.array(temp0)
  filter0 = numpy.zeros((10,10))
  W,H = temp0.shape[:2]
  filtersize = 6

  for i in range(W - filtersize):
    for j in range(H - filtersize):
      filter0 = temp1[i:i + filtersize,j:j + filtersize]

      flag = 0
      if sum(filter0[:,0]) == 0:
        flag +=1
      if sum(filter0[:,filtersize - 1]) == 0:
        flag +=1
      if sum(filter0[0,:]) == 0:
        flag +=1
      if sum(filter0[filtersize - 1,:]) == 0:
        flag +=1
      if flag > 3:
        temp2[i:i + filtersize, j:j + filtersize] = numpy.zeros((filtersize, filtersize))

  return temp2


def get_descriptors(img):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img = clahe.apply(img)
	img = image_enhance.image_enhance(img)
	img = numpy.array(img, dtype=numpy.uint8)
	# Threshold
	ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	# Normalize to 0 and 1 range
	img[img == 255] = 1

	#Thinning
	skeleton = skeletonize(img)
	skeleton = numpy.array(skeleton, dtype=numpy.uint8)
	skeleton = removedot(skeleton)
	# Harris corners
	harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
	harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
	threshold_harris = 125
	# Extract keypoints
	keypoints = []
	for x in range(0, harris_normalized.shape[0]):
		for y in range(0, harris_normalized.shape[1]):
			if harris_normalized[x][y] > threshold_harris:
				keypoints.append(cv2.KeyPoint(y, x, 1))
	# Define descriptor
	orb = cv2.ORB_create()
	# Compute descriptors
	_, des = orb.compute(img, keypoints)
	return (keypoints, des);

def get_keypoint_visualizations(testImage,matchedImage):
  kp1,des1 = get_descriptors(testImage)
  kp2,des2 = get_descriptors(matchedImage)
  
  img3 = cv2.drawKeypoints(testImage,kp1,outImage=None)
  img4 = cv2.drawKeypoints(matchedImage,kp2,outImage=None)
  
  f,axarr = plt.subplots(1,2)
  axarr[0].imshow(img3)
  axarr[1].imshow(img4)
  plt.show()
  
def get_matching_visualizations(testImage,matchedImage):
  kp1,des1 = get_descriptors(testImage)
  kp2,des2 = get_descriptors(matchedImage)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
  matches = sorted(bf.match(des1,des2),key= lambda match:match.distance)
  img = cv2.drawMatches(testImage,kp1,matchedImage,kp2,matches,flags=2,outImg=None)
  plt.imshow(img)
  plt.show()

def get_difference_score(img1,img2):
  kp1,des1 = get_descriptors(img1)
  kp2,des2 = get_descriptors(img2)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
  matches = sorted(bf.match(des1,des2),key= lambda match:match.distance)
  score = 0
  for match in matches:
    score += match.distance
  avg_score = score/len(matches)
  return avg_score

def test_finger(path,fingerType):
  print("Image path:",path,flush=True)
  for subject_index in range(0,50):
    print("Subject number:",subject_index,flush=True,end="\t\t")
    img1 = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("database/"+str(subject_index)+"_"+str(fingerType)+".jpg",cv2.IMREAD_GRAYSCALE)
    
    avg_score = get_difference_score(img1,img2)
    print("Score: ",avg_score,flush=True)
    if avg_score < SCORE_THRESHOLD:
      get_keypoint_visualizations(img1,img2)
      get_matching_visualizations(img1,img2)
      return subject_index
  return -2
  
# def main():
#   max_score=0
#   maxi,maxj = 0,0
#   fingers = ["thumb", "index", "middle", "ring", "little"]
  
#   #Comparing each finger with the corresponding finger in other subjects  
#   for finger_index in range(5):
#     aggregate_scores = []
#     print(fingers[finger_index].upper() + " FINGER")
#     for sub_index_1 in range(1,51):
#       max_difference_score = -1
#       difference_score_sum = 0
#       for sub_index_2 in range(sub_index_1,51):
#         img1 = cv2.imread("database/sub"+str(sub_index_1)+"/"+str(sub_index_1)+str(finger_index+1)+".jpg",cv2.IMREAD_GRAYSCALE)
#         img2 = cv2.imread("database/sub"+str(sub_index_2)+"/"+str(sub_index_2)+str(finger_index+1)+".jpg",cv2.IMREAD_GRAYSCALE)
        
#         avg_score = get_difference_score(img1,img2)
#         max_difference_score = max(max_difference_score,avg_score)
#         difference_score_sum += avg_score
        
#         print(sub_index_1,sub_index_2,avg_score,sep=" ")
#       avg_difference_score = difference_score_sum/50
#       aggregate_scores.append([max_difference_score,avg_difference_score])
#     print("Aggregate scores for "+fingers[finger_index]+" finger")
#     print(aggregate_scores)
#     print()
        
                
                
#Comparing each finger to other fingers in the same subject
# for k in range(1,51):
#     print("Sample no: "+str(k))
#     print("Max_i: "+str(maxi))
#     print("Max_j: "+str(maxj))
#     print("Max score:"+str(max_score))
#     for i in range(1,6):
#         for j in range(i,6):
#             img2 = cv2.imread("database/sub"+str(k)+"/"+str(k)+str(i)+".jpg",cv2.IMREAD_GRAYSCALE)
#             img1 = cv2.imread("database/sub"+str(k)+"/"+str(k)+str(j)+".jpg",cv2.IMREAD_GRAYSCALE)
#             kp1,des1 = get_descriptors(img1)
#             kp2,des2 = get_descriptors(img2)
#             bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
#             matches = sorted(bf.match(des1,des2),key= lambda match:match.distance)
#             score=0
#             for match in matches:
#     	        score += match.distance
#             avg_score = score/len(matches)
#             print(i,j,avg_score,sep="  ")
#             if (avg_score >= max_score):
#                 max_score = avg_score
#                 maxi = k*10 + i
#                 maxj = k*10 + j
#     print()


# if __name__ == "__main__":
# 	try:
# 		main()
# 	except:
# 		raise
