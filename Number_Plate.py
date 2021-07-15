import cv2
import os

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_russian_plate_number.xml')
numberplatecasc = cv2.CascadeClassifier(haar_model)
mina = 500
colour = (255,0,255)
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)

while True:
    success , img = cap.read()
    #img = cv2.imread(r'C:\Users\muska\Downloads\cars.jpg')
    # img = cv2.resize(img1, (200, 100))
    facecasc = cv2.CascadeClassifier(haar_model)
    imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    number_plates = numberplatecasc.detectMultiScale(img, 1.1, 4)

    for (x, y, w, h) in number_plates:
        area = w*h
        if area>mina:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img,"Number plate",(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,colour,2)
            imgres = img[y:y+h,x:x+w]
            cv2.imshow("Result",imgres)

    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF == ord('m'):
        break