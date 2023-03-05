from tqdm import tqdm
import numpy as np
import cv2

cv = cv2


cap = cv2.VideoCapture(0)
kernel = np.ones((5,5),np.uint8)

hsv_min = np.array((2, 28, 65), np.uint8)
hsv_max = np.array((26, 238, 255), np.uint8)

def main():
    while(True): 
        ret, frame = cap.read()
        edges= frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        

        """cv2.imshow('Original', frame)
        ret, th1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        gradient = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, kernel)
        gradient2 = cv2.adaptiveThreshold(cv2.cvtColor(gradient, cv2.COLOR_RGB2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

        cv2.imshow('Adaptive Gaussian Treshhold', th3)
        cv2.imshow('Gradient', gradient)
        cv2.imshow('Adaptive Gaussian Treshhold gradient', gradient2)
        """
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lsd = cv2.createLineSegmentDetector(0)
        lsd2 = cv2.createLineSegmentDetector(0)
        

        #Detect lines in the image
        lines = lsd.detect(gray)[0] #Position 0 of the returned tuple are the detected lines
        lines1 = cv2.Canny(	frame, 100, 240, edges,3) 

        #Draw detected lines in the image
        drawn_img = lsd.drawSegments(frame,lines)
        #drawn_img2 = lsd.drawSegments(frame,lines1)
        cv2.imshow('???', drawn_img)
        cv2.imshow('???!1', lines1)




        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()