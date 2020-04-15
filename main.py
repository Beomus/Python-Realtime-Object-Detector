from imutils.video import VideoStream, FPS
import imutils
import numpy as np
import time 
import cv2
import os

prototxt = 'MobileNetSSD_deploy.prototxt.txt'
model = 'MobileNetSSD_deploy.caffemodel'
CONFIDENCE = 0.2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS= np.random.uniform(0, 255, size=(len(CLASSES), 3))

print('[INFO] Loading model...')
model = cv2.dnn.readNetFromCaffe(prototxt, model)

print('[INFO] Loading camera...')
vs = VideoStream(scr=0).start()
time.sleep(2)
fps = FPS().start()

run = True
while run:
	frame = vs.read()
	# frame = imutils.resize(frame, width=600)
	
	h, w = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

	model.setInput(blob)
	detections = model.forward()

	for i in np.arange(0, detections.shape[2]):
		# extract the confidence associated with the detected object
		confidence = detections[0, 0, i, 1]
		
		if confidence > CONFIDENCE:
			index = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			x_start, y_start, x_end, y_end = box.astype('int')

			label = f'{CLASSES[index]}: {round(confidence * 100, 2)}'
			cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), COLORS[index], 2)
			y = y_start - 15 if y_start - 15 > 15 else y_start + 15
			cv2.putText(frame, label, (x_start, y), cv2.FONT_HERSHEY_PLAIN, 0.5, COLORS[index], 2)

	cv2.imshow('Frame', frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord('q'):
		run = False
		break

	fps.update()

fps.stop()
print(f'[INFO] Elasped time: {fps.elapsed()}')
print(f'[INFO] Average FPS: {fps.fps()}')

cv2.destroyAllWindows()
vs.stop()

