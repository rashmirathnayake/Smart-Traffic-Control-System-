# import the opencv library 
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import point_inside_polygon
import json
import socket



		 
# print ("Socket successfully created")
port = 12345



model = YOLO('yolov8n.pt')
vid1 = cv2.VideoCapture("footage/feed1.mp4") 
vid2 = cv2.VideoCapture("footage/feed2.mp4") 
vid3 = cv2.VideoCapture("footage/feed3.mp4") 
vid4 = cv2.VideoCapture("footage/feed3.mp4") 

polys = [np.array([[166, 247], [384, 239], [523, 417], [48, 415],[166, 247]],np.int32),
		 np.array([[262, 232], [458, 232], [603, 422], [109, 418],[262, 232]],np.int32),
		 np.array([[188, 239], [299, 240], [271, 362], [2, 340],[188, 239]],np.int32),
		np.array([[344, 242], [466, 245], [639, 343], [363, 355],[344, 242]],np.int32)]


# reshaped_polys = [np.array([[361, 377], [169, 637], [949, 629], [725, 373]],np.int32),
# 		 np.array([[504, 341], [232, 617], [1076, 601], [856, 341]],np.int32),
# 		 np.array([[358, 357], [566, 357], [498, 613], [14, 569]],np.int32),
# 		np.array([[648, 359], [868, 352], [1213, 578], [700, 614]],np.int32)]

# for x in range(len(polys)):
# 	reshaped_polys[x]=polys[x].reshape((-1,1,2))


vid = [vid1 , vid2, vid3, vid4]
	 



    # Establish connection with client. 

# send a thank you message to the client. encoding to send byte type. 
# c.send('Thank you for connecting'.encode()) 



def getcount(vid,polys):
    i =0
    annotated_frame = []
    firstloop=True
    ret =True
    vehicles=[0,0,0,0]
    while(ret): 
        
        # Capture the video frame 


        ret, frame = vid[i].read() 
        ret, frame = vid[i].read() 
        ret, frame = vid[i].read() 
        ret, frame = vid[i].read() 

        results = model.predict(frame,classes=[2, 3, 5, 7], save=False,device=0, tracker="bytetrack.yaml",verbose = False,show=False)


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
        
        vehicles[i]=count
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            print("end reached!")
            break

        if i==3:
            
            return json.dumps({"count":
                
                    [{   "lane":"upper",
                        "count":vehicles[0]
                    },
                    {   "lane":"right",
                        "count":vehicles[1]
                    },
                    {   "lane":"down",
                        "count":vehicles[2]
                    },
                    {"  lane":"left",
                        "count":vehicles[3]
                    }]
                ,
                    "status":"running"})
            
        else:
            i+=1
	


# After the loop release the cap object 

print("server running ")
response = "get"
while True:
    
    
    s = socket.socket()
    s.bind(('', port))
    s.listen(5)
    c, addr = s.accept()

    response = c.recv(1024).decode()
    print("Received from server:", response)
    # Check if the stop signal is received
    if response.lower() == "stop":
        print("Received stop signal. Exiting.")
        c.send(json.dumps({"count":
                
                   [ {   "lane":"upper",
                        "count":0
                    },
                    {"lane":"right",
                        "count":0
                    },
                    {"lane":"down",
                        "count":0
                    },
                    {"lane":"left",
                        "count":0
                    }]
                ,
                    "status":"terminated"}).encode())
        c.close()
        break
    else:
        c.send(getcount(vid,polys).encode())

vid1.release() 
vid2.release() 
vid3.release() 
vid4.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
