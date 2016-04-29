# the original sources could be find here
# https://rahullpatell.wordpress.com/2015/04/21/real-time-face-detection-using-viola-jones-and-camshift-in-python-i/

from face.facedetector import FaceDetector
import cv2
import numpy as np

# set the RATIO by which we want to resize the image
RATIO = 2
# number of frames we want to track the face after detection
TRACK = 30
# number of frames to be skipped if no face is found
SKIP = 2

cap = cv2.VideoCapture(0)

termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

def VJFindFace(frame):
    global RATIO, orig
    # list to store the coordinates of the faces
    allRoiPts = []    
    # generate a copy of the original frame
    orig = frame.copy()    
    # resize the original image
    dim = (frame.shape[1]/RATIO, frame.shape[0]/RATIO);
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)                
    # convert the frame to gray scale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)        
    # find faces in the gray scale frame of the video using Haar feature based trained classifier
    fd = FaceDetector('cascades/haarcascade_frontalface_default.xml')
    faceRects = fd.detect(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (10, 10))    
    # loop over the faces and draw a rectangle around each
    for (x, y, w, h) in faceRects:
        # decrease the size of the bounding box
        x = RATIO*(x+10)
        y = RATIO*(y+10)
        w = RATIO*(w-15)
        h = RATIO*(h-15)            
        
        # original result of Viola-Jones algorithm
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # insert the coordinates of each face to the list
        allRoiPts.append((x, y, x+w, y+h))        
    
    # show the detected faces
    cv2.imshow("Faces", frame)
    cv2.waitKey(1)  
    return allRoiPts

def trackFace(allRoiPts, allRoiHist):        
    for k in range(0, TRACK):
        # read the frame and check if the frame has been read properly
        ret, frame = cap.read()
        if not ret:
            return -1;
            break
        i=0
        # convert the given frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # for histogram of each window found, back project them on the current frame and track using CAMSHIFT
        for roiHist in allRoiHist:
            backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
            (r, allRoiPts[i]) = cv2.CamShift(backProj, allRoiPts[i], termination)  
            # error handling for bound exceeding
            for j in range(0,4):         
                if allRoiPts[i][j] < 0:
                    allRoiPts[i][j] = 0
            pts = np.int0(cv2.cv.BoxPoints(r))        
            # draw bounding box around the new position of the object
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
            i = i + 1            
        # show the face on the frame
        cv2.imshow("Faces", frame)
        cv2.waitKey(1)
    return 1;

def calHist(allRoiPts):
    global orig
    allRoiHist = []    
    # convert each face to HSV and calculate the histogram of the region
    for roiPts in allRoiPts:
        roi = orig[roiPts[1]:roiPts[-1], roiPts[0]:roiPts[2]]            
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
        roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
        allRoiHist.append(roiHist);
    return allRoiHist
        
def justShow():
    global cap,SKIP
    # read and display the frame
    for k in range(0,SKIP):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Faces", frame)
        cv2.waitKey(1)

def main():
    global cap
    i=0
    while(cap.isOpened()):                
        # try to find faces using Viola-Jones algorithm
        # if faces are found, track them
        # if no faces are found, don't search for faces for the next 2 frames
        # repeat until a face has been found
        if i % 2 == 0:
            # erase the pervious faces and their hsv histograms before each call
            allRoiPts = []
            allRoiHist = []
            # read the frame
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cv2.destroyAllWindows()
                return                
            
            allRoiPts = VJFindFace(frame)
                                        
            # check if any faces have been found
            if len(allRoiPts) != 0:
                allRoiHist = calHist(allRoiPts)
                i=i+1
            else:
                # skip the next 2 frames if no faces have been found
                justShow()

        else:
            # track the faces if any have been found
            error = trackFace(allRoiPts, allRoiHist)
            if error == -1:
                cap.release()
                cv2.destroyAllWindows()
                return
            i=i+1                

        # press 'q' to exit the script
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
