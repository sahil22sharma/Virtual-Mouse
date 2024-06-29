import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
import pyautogui
import numpy as np

##################
wCam,hCam = 640,480
frameR = 100
smoothening = 5
##################

plocX , plocY = 0 ,0
clocX , clocY = 0 ,0

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime= 0
detector = htm.handDetector(maxHands=1)
wScreen, hScreen = pyautogui.size()
# print(wScreen)
# print(hScreen)
while True:
    #1. Find hand landmarks
    success,img = cap.read()
    img = detector.findHands(img)
    lmList, bbox =detector.findPosition(img)

    #2. Get the tip of the index and middle fingers
    if len(lmList)!=0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]

        #3.Check which fingers are up
        fingers = detector.fingersUp()

        #4.Only index finger : Moving mode
        if fingers[1]==1 and fingers[2]==0:
            cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(225,0,225),2)
            #5. Convert Coordinates
            x3=np.interp(x1,(frameR,wCam-frameR),(0,wScreen))
            y3=np.interp(y1,(frameR,hCam-frameR),(0,hScreen))

            #6. Smoothen the values
            clocX = plocX+(x3-plocX)/smoothening
            clocY = plocY+(y3-plocY)/smoothening
            #7. Move Mouse
            pyautogui.moveTo(wScreen-clocX,clocY)
            cv2.circle(img,(x1,y1),10,(225,0,225),cv2.FILLED)
            plocX,plocY=clocX,clocY

        #8. Both index and middle fingers are up : Clicking mode
        if fingers[1]==1 and fingers[2]==1:
            #9. FInd distance between fingers
            length ,img, lineInfo = detector.findDistance(8,12,img)
            # print(length)
            #10. Click mouse if distance short
            if length<25:
                cv2.circle(img,(lineInfo[4],lineInfo[5]),5,(0,225,0),cv2.FILLED)
                pyautogui.click()
                
    #11. Frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(20,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    #12. Display
    cv2.imshow("Image",img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
