import numpy as np
import cv2
import time
                                    ########################
                                    #### INITIALIZATION ####
                                    ########################

# modify to read from a txt file

# marker for dictionary
mark1   = [ [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [0, 0, 1, 1, 0], [1, 1, 1, 0, 1] ]
mark2   = [ [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 1, 0], [1, 0, 1, 1, 0] ]
mark3   = [ [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [0, 1, 1, 1, 1], [1, 0, 1, 0, 0] ]
mark4   = [ [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0] ]
mark5   = [ [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 1, 1], [0, 1, 1, 0, 0] ]
mark6   = [ [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 1, 1] ]
mark7   = [ [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 0], [0, 0, 1, 0, 1] ]
mark8   = [ [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [0, 0, 1, 0, 1], [1, 1, 1, 1, 0] ]
mark9   = [ [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 0, 0], [1, 1, 1, 0, 0] ]
mark10  = [ [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [0, 1, 1, 0, 0], [1, 0, 1, 1, 1] ]
mark11  = [ [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1] ]
mark12  = [ [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [0, 1, 1, 1, 1] ]
mark13  = [ [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [0, 1, 1, 0, 1], [0, 1, 1, 0, 1] ]
mark14  = [ [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 0, 1], [0, 0, 1, 1, 0] ]
mark15  = [ [1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 0, 1, 0, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0] ]

MARKER_ARRAY = [mark1, mark2, mark3, mark4, mark5, mark6, mark7, mark8, mark9, mark10, mark11, mark12, mark13, mark14, mark15]
MARK_NUMBER = len(MARKER_ARRAY)
DEBUG = True


                                    ####################
                                    #### SOME CHECK ####
                                    #####################


# I used this piece of code to check the shape of a predefined
# aruco dictionary with marker 5x5 to initialize correctly
# my custom dictionary. I get lots of error  

# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_dict_try = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

if DEBUG == True:
    print("An aruco_dict.byte_list: ")

    print("shape: {}".format(aruco_dict_try.bytesList.shape) )
    print("type bytelist: {}".format( type(aruco_dict_try.bytesList)) )
    print("type bytelist[0]: {}".format( type(aruco_dict_try.bytesList[0]) ) )
    print(aruco_dict_try.bytesList[0])

                                    #######################
                                    #### START OF CODE ####
                                    #######################

# define an empty custom dictionary:
#   - first parameter:   number of image's chanel ( ? )
#   - secondo parameter: number of bit of aruco marker
#   - third parameter:   ( ? )

aruco_dict = cv2.aruco.custom_dictionary(3, 5, 1)


# add empty bytesList which contain MARKER_NUMBER marker
aruco_dict.bytesList = np.empty(shape = (MARK_NUMBER, 4, 4), dtype = np.uint8)

# adding all marker into dictionary
for i in range(len(MARKER_ARRAY)):
    
    mark = MARKER_ARRAY[i]

    mybits = np.array(mark, dtype = np.uint8)
    aruco_dict.bytesList[i] = cv2.aruco.Dictionary_getByteListFromBits(mybits)

# save marker images
for i in range(len(aruco_dict.bytesList)):
    cv2.imwrite("marker_drawn/custom_aruco_" + str(i+1) + ".png", cv2.aruco.drawMarker(aruco_dict, i, 128, borderBits=2 ) )

print(aruco_dict.bytesList)
print("-- dictionary created")


#####################




                                    ########################
                                    #### TEST DETECTION ####
                                    ########################


# function for draw line around the marker
def aruco_display(corners, ids, rejected, image, old_id=0):
	if len(corners) > 0:
		
		ids = ids.flatten()
		
		for (markerCorner, markerID) in zip(corners, ids):
			
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			
			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)

			if old_id != ids:
				print("[Inference] ArUco marker ID: {}".format(markerID))
			
	return image


# test detection of aruco marker on an image
img = cv2.imread("marker_drawn/cube_aruco.png", cv2.IMREAD_COLOR)
corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict)
detected_markers = aruco_display(corners, ids, rejected, img)


cv2.imshow("Image", detected_markers)
#cv2.waitKey(0)     # uncomment for test and comment the next lines of code


# if you uncomment, you can see that, using another dicitonary, donot' drawn any mark
#aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)


cap = cv2.VideoCapture("marker_video/mars3.ogv")
#cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

old_id = 0
while cap.isOpened():
    
	ret, img = cap.read()

	h, w, _ = img.shape

	width = 1000
	height = int(width*(h/w))
	img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
	corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict)

	detected_markers = aruco_display(corners, ids, rejected, img, old_id)
	old_id = ids

	#cv::aruco::drawDetectedMarkers (InputOutputArray image, InputArrayOfArrays corners, InputArray ids=noArray(), Scalar borderColor=Scalar(0, 255, 0))

	cv2.aruco.drawDetectedMarkers(detected_markers, corners, ids )

	cv2.imshow("Image", detected_markers)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
	    break

cv2.destroyAllWindows()
cap.release()

