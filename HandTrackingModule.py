import cv2
import mediapipe as mp
import time
import math

# class creation
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils # it gives small dots onhands total 20 landmark points
        self.tipIds=[4,8,12,16,20]

    def findHands(self,img,draw=True):
        # Send rgb image to hands
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(cv2.flip(img,1),cv2.COLOR_BGR2RGB) # for flipping the image 
        self.results = self.hands.process(imgRGB) # process the frame
        # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) # for flipping the image
    #     print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    #Draw dots and connect them
                    self.mpDraw.draw_landmarks(img,handLms,
                                                self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self,img, handNo=0, draw=True):
        """Lists the position/type of landmarks
        we give in the list and in the list ww have stored
        type and position of the landmarks.
        List has all the lm position"""
        xList=[]
        yList=[]
        bbox=[]
        self.lmlist = []

        # check wether any landmark was detected
        if self.results.multi_hand_landmarks:
            #Which hand are we talking about
            myHand = self.results.multi_hand_landmarks[handNo]
            # Get id number and landmark information
            for id, lm in enumerate(myHand.landmark):
                # id will give id of landmark in exact index number
                # height width and channel
                h,w,c = img.shape
                #find the position
                cx,cy = int(lm.x*w), int(lm.y*h) #center
                # print(id,cx,cy)
                xList.append(cx)
                yList.append(cy)
                self.lmlist.append([id,cx,cy])

                # Draw circle for 0th landmark
                if draw:
                    cv2.circle(img,(cx,cy), 5 , (225,0,225), cv2.FILLED)
            xmin,xmax = min(xList),max(xList)
            ymin,ymax = min(yList),max(yList)
            bbox = xmin,ymin,xmax,ymax
            if draw:
                cv2.rectangle(img,(bbox[0]-15,bbox[1]-15),(bbox[2]+15,bbox[3]+15),(0,225,0),2)
        return self.lmlist, bbox
    
    def fingersUp(self):
        fingers=[]
        #thumb
        if self.lmlist[self.tipIds[0]][1]>self.lmlist[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        #fingers
        for id in range(1,5):
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    
    def findDistance(self,p1,p2,img,draw=True):
       
        x1,y1=self.lmlist[p1][1],self.lmlist[p1][2]
        x2,y2=self.lmlist[p2][1],self.lmlist[p2][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2

        if draw:
            cv2.circle(img,(x1,y1),10,(225,0,225),cv2.FILLED)
            cv2.circle(img,(x2,y2),10,(225,0,225),cv2.FILLED)
            cv2.line(img,(x1,y1),(x2,y2),(0,0,225),3)
            cv2.circle(img,(cx,cy),5,(0,0,225),cv2.FILLED)

        length = math.hypot(x2-x1,y2-y1)
        return length,img,[x1,y1,x2,y2,cx,cy]

def main():
    #Frame rates
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success,img = cap.read()
        img = detector.findHands(img)#give an argument draw = False for removing the default drawing made on your hand while tracking 
        lmList = detector.findPosition(img)#give an argument draw = False for removing the custom drawing
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

        cv2.imshow("Image",img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()