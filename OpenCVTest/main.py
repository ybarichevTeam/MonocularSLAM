#!/usr/bin/env python3



from tqdm import tqdm
import numpy as np
import cv2

cv = cv2


cap = cv2.VideoCapture(0)
kernel = np.ones((5,5),np.uint8)



def main():
    files = ["./src/коридор.jpg"]# , "./src/поле.jpg"]
    for f in files:
        img = cv2.imread(f, 1) 
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img=gray
        # Roberts
        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])
        img_robertx = cv2.filter2D(img, -1, kernelx)
        img_roberty = cv2.filter2D(img, -1, kernely)
        roberts = cv2.addWeighted(img_robertx, 0.5, img_roberty, 0.5, 0)

        # Prewitt
        kernelx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        img_robertx = cv2.filter2D(img, -1, kernelx)
        img_roberty = cv2.filter2D(img, -1, kernely)
        prewitt = cv2.addWeighted(img_robertx, 0.5, img_roberty, 0.5, 0)

        # Sobel
        kernelx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        kernely = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        img_robertx = cv2.filter2D(img, -1, kernelx)
        img_roberty = cv2.filter2D(img, -1, kernely)
        sobel = cv2.addWeighted(img_robertx, 0.5, img_roberty, 0.5, 0)
        #sobel2 = cv2.Sobel(gray, cv2.CV_32F, 1, 0)

        # Canny
        # Setting parameter values
        t_lower = 50  # Lower Threshold
        t_upper = 150  # Upper threshold
        
        # Applying the Canny Edge filter
        canny = cv2.Canny(img, t_lower, t_upper)
        canny2 = cv2.Canny(img, t_lower, t_upper, L2gradient = True )

        #Laplace
        ddepth = cv.CV_16S
        kernel_size = 3
        lapl = cv.convertScaleAbs(cv.Laplacian(gray, ddepth, ksize=kernel_size))
        
        
        #kirsh
        kernel1 = np.array([[5,5,5], [-3, 0,-3], [-3,-3,-3]])
        kernel2 = np.array([[5,5,-3], [5, 0,-3], [-3,-3,-3]])
        kernel3 = np.array([[5,-3,-3], [5, 0,-3], [5,-3,-3]])
        kernel4 = np.array([[5,-3,-3], [5, 0,-3], [5,-3,-3]])
        kernel5 = np.array([[-3,-3,-3], [5, 0,-3], [5,5,-3]])
        
        koeff = 1/9
        img_robert1 = cv2.filter2D(img, -1, kernel1)
        img_robert2 = cv2.filter2D(img, -1, kernel2)
        img_robert3 = cv2.filter2D(img, -1, kernel3)
        img_robert4 = cv2.filter2D(img, -1, kernel4)
        kirsh = cv.addWeighted(img_robert1,koeff, img_robert2,koeff,0)
        kirsh = cv.addWeighted(kirsh,2*koeff, img_robert3,koeff,0)
        kirsh = cv.addWeighted(kirsh,3*koeff, img_robert4,koeff,0)



        cv2.imshow('Roberts', cv.threshold(roberts,20,255,cv.THRESH_BINARY)[1])
        cv2.imshow('Prewitt', cv.threshold(prewitt,50,255,cv.THRESH_BINARY)[1])
        cv2.imshow('Sobel', cv.threshold(sobel,60,255,cv.THRESH_BINARY)[1])
        #cv2.imshow('Sobel2', sobel2)
        #cv2.imshow('Canny', canny)
        cv2.imshow('Canny2', canny2)
        cv2.imshow('Laplace', cv.threshold(sobel,100,255,cv.THRESH_BINARY)[1])
        cv2.imshow('Kirsh', kirsh)
        # cv2.imshow('???!1', th1)"""



    while (True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()