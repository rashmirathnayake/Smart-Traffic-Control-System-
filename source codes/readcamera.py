# import the opencv library 
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import point_inside_polygon



# s = socket.socket()		 
# print ("Socket successfully created")
# port = 12345	
# s.bind(('', port))		 
# print ("socket binded to %s" %(port)) 


model = YOLO('yolov8n.pt')
vid1 = cv2.VideoCapture("footage/feed1.mp4") 
vid2 = cv2.VideoCapture("footage/feed2.mp4") 
vid3 = cv2.VideoCapture("footage/feed3.mp4") 
vid4 = cv2.VideoCapture("footage/feed3.mp4") 

polys = [np.array([[166, 247], [384, 239], [523, 417], [48, 415],[166, 247]],np.int32),
		 np.array([[262, 232], [458, 232], [603, 422], [109, 418],[262, 232]],np.int32),
		 np.array([[188, 239], [299, 240], [271, 362], [2, 340],[188, 239]],np.int32),
		np.array([[344, 242], [466, 245], [639, 343], [363, 355],[344, 242]],np.int32)]


reshaped_polys = [np.array([[361, 377], [169, 637], [949, 629], [725, 373]],np.int32),
		 np.array([[504, 341], [232, 617], [1076, 601], [856, 341]],np.int32),
		 np.array([[358, 357], [566, 357], [498, 613], [14, 569]],np.int32),
		np.array([[648, 359], [868, 352], [1213, 578], [700, 614]],np.int32)]

for x in range(len(polys)):
	reshaped_polys[x]=polys[x].reshape((-1,1,2))


vid = [vid1 , vid2, vid3, vid4]
i =0
# s.listen(5)	 
# print ("socket is listening")	



    # Establish connection with client. 
# c, addr = s.accept()	 
# print ('Got connection from', addr )

# send a thank you message to the client. encoding to send byte type. 
# c.send('Thank you for connecting'.encode()) 

annotated_frame = []
firstloop=True
ret =True
while(ret): 
	
	# Capture the video frame 


	ret, frame = vid[i].read() 
	ret, frame = vid[i].read() 
	ret, frame = vid[i].read() 
	#ret, frame = vid[i].read() 

	results = model.predict(frame,classes=[2, 3, 5, 7], save=False,device=0, tracker="bytetrack.yaml",verbose = False)


	if(firstloop):
		annotated_frame = [ results[0].plot(), results[0].plot(), results[0].plot(), results[0].plot()]
		firstloop=False
	else:
		annotated_frame[i] = results[0].plot()

	
	boxes = results[0].boxes.xywh.cpu()
	count=0
	for box in boxes:
		x,y,h,w = box

		if (point_inside_polygon.point_inside_polygon((x,y),polys[i])):
			count+=1
	
	cv2.polylines(annotated_frame[i],[reshaped_polys[i]],False,(0,255,0),2)
	# Display the resulting frame 
	count_text = f"Vehicle count: {count}"
	cv2.putText(annotated_frame[i], count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

	row1=cv2.hconcat([annotated_frame[0],annotated_frame[1]])
	row2=cv2.hconcat([annotated_frame[2],annotated_frame[3]])

	combined_frame = cv2.vconcat([row1,row2])
	cv2.imshow('frame', combined_frame) 
	# c.send(str(count).encode())
	# the 'q' button is set as the 
	# quitting button you may use any 
	# desired button of your choice 
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		print("end reached!")
		break

	if i==3:
		i=0
	else:
		i+=1
	
# c.close()
# After the loop release the cap object 
vid1.release() 
vid2.release() 
vid3.release() 
vid4.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
