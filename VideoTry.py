import cv2
cap = cv2.VideoCapture(2)
ret,frame = cap.read()
while True:
        ret, frame = cap.read()
        blurred = cv2.GaussianBlur(frame,(3,3),0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        #cv2.putText(frame, f'Laplacian variance: {laplacian_var}', (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if laplacian_var > 14 and laplacian_var < 35:
            cv2.putText(frame,"The image is Rough",(10,450),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2)
        elif laplacian_var > 35:
            cv2.putText(frame,"The image is Uneven",(10,450),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        else:
            cv2.putText(frame,"The image is Smooth",(10,450),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
        cv2.imshow('Contours', frame)
        if cv2.waitKey(1) == ord('q'):
             break
ImgPath.release()
cv2.destroyAllWindows()

	
