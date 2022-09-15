import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('/home/yashas/catkin_ws/src/img_to_xyz/src/face_detect_cascade.xml') #Change the File path (too lzy to edit cmakelists :) )

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    print(faces)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    try:
        center_x = (x+(x+w))//2
        center_y = (y+(y+h))//2
    except:
        center_x = 0
        center_y = 0
    for (x,y,w,h) in faces :
        cv2.circle(img,(center_x,center_y),5,(255, 0, 0), 2)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()