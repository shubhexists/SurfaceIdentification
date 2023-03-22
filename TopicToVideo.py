import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
import cv2 
def callback(data):
  br = CvBridge()
  rospy.loginfo("receiving video frame")
  current_frame = br.imgmsg_to_cv2(data)
  blurred = cv2.GaussianBlur(current_frame,(3,3),0)
  gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(current_frame, contours, -1, (0, 255, 0), 2)
  laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
  #cv2.putText(frame, f'Laplacian variance: {laplacian_var}', (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  if laplacian_var > 14 and laplacian_var < 35:
        cv2.putText(current_frame,"The image is Rough",(10,450),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2)
  elif laplacian_var > 35:
        cv2.putText(current_frame,"The image is Uneven",(10,450),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
  else:
        cv2.putText(current_frame,"The image is Smooth",(10,450),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
  cv2.imshow('Contours', current_frame)
  cv2.waitKey(1)
  #cv2.imshow("camera", current_frame)
  #cv2.waitKey(1)
def receive_message():
  rospy.init_node('video_sub_py', anonymous=True)
  rospy.Subscriber('/usb_cam1/image_raw', Image, callback)
  rospy.spin()
  cv2.destroyAllWindows()
  
if __name__ == '__main__':
  receive_message()
