import cv2
import numpy as np
from object_detection import ObjectDetection #modelul de pe YouTube
import math

# Load/Initialize the Object Detection
od = ObjectDetection()

# 1) load the video footage

cap = cv2.VideoCapture(r"C:\\Users\sorina.oancea\PycharmProjects\objectTrackingProject\source-code-Pysource\los_angeles.mp4")

# count frames
count = 0

center_points_prev_frame = [] #daca nu initializam cu 0 crapa deoarece nu exista prev. pentru primul frame

tracking_objects = {}
track_id = 0

#center_points = [] #on each loop we will put new points in array

#if the video has 100 frames we do not need to keep the history for all of that layers
#we compare just CURRENT FRAME with NEW FRAME - we work with center points - center_points

# 2) get the frames from the video

            # _, frame = cap.read() # one frame from the video
            #
            # cv2.imshow("Frame", frame)
            # cv2.waitKey(0) #a function which keeps the window open -> it freezez the frame

# get the frames one after another -> put in a loop

while True:
    # _, frame = cap.read() - mergea ok dar este o eroare cand aplica YOLO4 pe un frame care nu exista
    ret, frame = cap.read()
    count += 1  #count++ cand mai gaseste un frame
    if not ret:
        break

    #get center points of current frame center_points
    center_points_cur_frame = []

    #detect the objects on the frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        #print(box) -> Array cu x,y,z,h - coordonate, inaltime
        (x, y, w, h) = box

        #initial era doar detectia-pentru tracking trebuie sa "urmarim" obiectul, dintr-un frame in altul
        cx = int((x + x + w) / 2)   #cx - center point
        cy = int((y + y + w) / 2)
        center_points_cur_frame.append((cx, cy)) #le memoram in vector

        #cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1) #desen cerc pentru marcarea center point
        print("Frame number", count, " ", x, y, w, h) #frame number si coordonatele
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) #(0, 255, 0) RGB


    #we will do the comparation just for the first frame, in order to keep the IDs for the objects: for first frame we compare the first and second frame
    if count <= 2: #we need to go a couple of frames to get the tracking object
    #in the loop we can loop to current position-prev. position, compare the distance and if the points are really close
    #we can say that it is the same object, we assign an id and we start the tracking
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                #check the distance - can do it with math library
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1]) #x si y

                if distance < 20: #10 pixeli - 20 pentru ca sa identifice toate obiectele
                    #creem un id special pentru acel obiect
                    tracking_objects[track_id] = pt #memoram center point in colectia de tracking
                    track_id += 1
    else:
        #compare with the central points we already have
        #mai intai for pentru dictionar ca sa urmarim mai intai obiectele -> erau inversate iar id-ul era pe ecran

        #make a copy for dictionary because the error: RuntimeError: dictionary changed size during iteration
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy() #in order to reuse the index

        for object_id, pt2 in tracking_objects_copy.items():
            #check to see if the object it's still on the frame or not
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1]) #calculate the distance

                if distance < 20:
                    #update the position, keep the same tracking
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue #to save some resources if you found the object
            #remove id if object does not exist in the frame - Lost id
            if not object_exists:
                tracking_objects.pop(object_id)
        for pt in center_points_cur_frame:
            #to encrease the lenght of the dictionary we need to get the lenght of it - if we see more objects in the frame
            tracking_objects[track_id] = pt
            track_id += 1


    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    print("Tracking objects")
    print(tracking_objects)

    print("Current Frame left points")
    print(center_points_cur_frame)


        #cv2.circle(frame, pt, 5, (0, 0, 255), -1) #desena cercul in centrul detectiei

    #PRINTARE CENTER POINTS PENTRU CURRENT FRAME AND PREV FRAME
    print("Current Frame")
    print(center_points_cur_frame)
    print("Prev Frame")
    print(center_points_prev_frame)
    cv2.imshow("Frame", frame)


        #add the central point and store the position in real time, so when we go to the new frames we do not lose the position
        #for prev. points -> for tracking

    cv2.imshow("Frame",frame)
    #key = cv2.waitKey(1) #wait 1ms between each frame
    key = cv2.waitKey(0) #freezez the frame - cu 0

    #before ending the loop we need a copy - because we compare 2 frames
    center_points_prev_frame = center_points_cur_frame.copy()

    #we need to check the points(for distance) between the center - in 2 frames: current and prev.

    if key == 27: # 27 = s key on the keyboard
        break

cap.release()
cv2.destroyAllWindows()