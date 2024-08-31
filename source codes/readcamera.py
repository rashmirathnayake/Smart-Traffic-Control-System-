# import the opencv library 
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import point_inside_polygon

import socket			 


s = socket.socket()		 
print ("Socket successfully created")
port = 12345	
s.bind(('', port))		 
print ("socket binded to %s" %(port)) 


# define a video capture object 
model = YOLO('yolov8n.pt')
vid = cv2.VideoCapture("footage/video1.mp4") 
s.listen(5)	 
print ("socket is listening")	



    # Establish connection with client. 
c, addr = s.accept()	 
print ('Got connection from', addr )

# send a thank you message to the client. encoding to send byte type. 
c.send('Thank you for connecting'.encode()) 

ret =True
while(ret): 
	
	# Capture the video frame 
	# by frame 
	ret, frame = vid.read() 
	results = model.predict(frame,classes=[2, 3, 5, 7], save=False,device=0, tracker="bytetrack.yaml",verbose = False)
	#detections = sv.Detections.from_ultralytics(results[0])
	annotated_frame = results[0].plot()
	poly =np.array([[429, 336],[168, 674],[919, 623],[712, 326],[429, 336]],np.int32)
	boxes = results[0].boxes.xywh.cpu()
	count=0
	for box in boxes:
		x,y,h,w = box

		if (point_inside_polygon.point_inside_polygon((x,y),poly)):
			count+=1
	poly=poly.reshape((-1,1,2))
	cv2.polylines(annotated_frame,[poly],False,(0,255,0),2)
	# Display the resulting frame 
	count_text = f"Vehicle count: {count}"
	cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	cv2.imshow('frame', annotated_frame) 
	c.send(str(count).encode())
	# the 'q' button is set as the 
	# quitting button you may use any 
	# desired button of your choice 
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		print("end reached!")
		break
c.close()
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
