# Code modified from: https://docs.opencv.org/3.4.0/db/d5b/tutorial_py_mouse_handling.html
import numpy as np
import cv2 as cv
from Common import *
import sys

drawing = False 

# Create canvas
canvas = np.zeros((512,512,1), np.uint8)
canvas[:] = (0)

def classify(img, networkModel):
	resizedImage = cv.resize(img, dsize=(28, 28))
	resizedImage = np.reshape(resizedImage, [1] + list(resizedImage.shape) + [1])
	resizedImage = resizedImage.astype('float32')
	resizedImage = preprocessImages(resizedImage, None)
	pred = networkModel.predict(resizedImage)
	digit = np.argmax(pred[0])
	return str(digit)

# mouse callback function
def draw_circle(event, x, y, flags, param):
	global drawing, canvas
	
	if event == cv.EVENT_LBUTTONDOWN:
		drawing = True        
	elif event == cv.EVENT_MOUSEMOVE:
		if drawing == True:
			cv.circle(canvas,(x,y),20,(255,255,255),-1)
	elif event == cv.EVENT_LBUTTONUP:
		drawing = False

def main():
	_, NetworkName = parseArgs()

	print("Loading", NetworkName)

	# Load model
	#networkModel = loadModelAndWeights(NetworkName)
	networkModel = loadModel(NetworkName)
		
	# Create window
	windowName = "Canvas"
	cv.namedWindow(windowName)

	# Set mouse callback
	cv.setMouseCallback(windowName,draw_circle)

	# While forever...
	while(1):
		# Show image
		cv.imshow(windowName, canvas)

		# Get key press
		k = cv.waitKey(1) & 0xFF		
		if chr(k) == 'c':
			print("Clear")
			canvas[:] = (0)
		elif chr(k) == 'f':
			print("DIGIT:", classify(canvas, networkModel))
		elif k == 27:
			break
	cv.destroyAllWindows()

if __name__ == "__main__": main()
