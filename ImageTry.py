import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# testing_image = cv2.imread('/home/jerry/Desktop/SurfaceDetection/road2.jpg')
# testing_image_gray = cv2.cvtColor(testing_image, cv2.COLOR_BGR2GRAY)
# figure = plt.figure(figsize=(10,10))
# figure.add_subplot(1,2,1)
# plt.imshow(testing_image_gray,cmap='gray')
# testing_image_gray_inverted = cv2.bitwise_not(testing_image_gray)
# ret, thresh = cv2.threshold(testing_image_gray_inverted, 127, 255,cv2.THRESH_BINARY)
# figure.add_subplot(1,2,2)
# plt.imshow(thresh,cmap='gray')
# plt.show(figure)

def FindContours(ImgPath):
    img = cv2.imread(ImgPath)
    # height, width, channels = img.shape
    # print("Image dimensions: {} x {} pixels".format(width, height))
    resized_img = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    img_inverted_gray = cv2.bitwise_not(img_gray)
    g_blurred_image = cv2.blur(img_inverted_gray,(6,6),0)
    ret, thresh = cv2.threshold(g_blurred_image, 150, 255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print("Kushagra Bhaiya is Image me ",len(contours)," Contours hai.")
    return contours

def DrawContours(ImgPath):
    img = cv2.imread(ImgPath)
    resized_img = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))
    figure = plt.figure(figsize=(20,10))
    figure.add_subplot(1,2,1)
    plt.imshow(resized_img[:,:,::-1])
    contours = FindContours(ImgPath)
    RGB_img = cv2.cvtColor(resized_img,cv2.COLOR_BGR2RGB)
    cv2.drawContours(RGB_img, contours, -1, (0,255,0), 2) #5 is the thickness of the line 
    figure.add_subplot(1,2,2)
    plt.imshow(RGB_img)
    plt.show(figure)
# DrawContours('/home/jerry/Desktop/SurfaceDetection/mars2.webp')
#-----------------------------------------------------------------------------------------------------------------------
#Half Scaling of Image , third scaling of image
#rough1.jpg = 124 , 89
#soil.png = 201 , 129
#road1.jpg = 9 , 3 
#mars1.jpg = 1520 , 760
#asliGround.jpg = 304 , 167
#mars2.webp = 689 , 462
#ground.jpg = 552 , 308
#rocky.png = 187 , 62
#------------------------------------------------------------------------------------------------------------------------
x_coordinates = []
y_coordinates = []
def ContourCoordinates(ImgPath):
    img = cv2.imread(ImgPath)
    # resized_img = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))
    img_contours = FindContours(ImgPath)
    for i in range(len(img_contours)):
        x, y, w, h = cv2.boundingRect(img_contours[i])
        x_coordinates.append(x)
        y_coordinates.append(y)
    return [x_coordinates, y_coordinates]


def plotDistributionCurve(ImgPath):
    img = cv2.imread(ImgPath)
    # resized_img = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))
    x,y = ContourCoordinates(ImgPath)
    # figure = plt.figure(figsize=(20,10))
    hist_x, bins_x = np.histogram(x,density=True)
    bin_centers_X = (bins_x[1:] + bins_x[:-1]) / 2
    # figure.add_subplot(1,2,1)
    # plt.plot(bin_centers_X, hist_x, '-', color='black')
    # plt.fill_between(bin_centers_X, hist_x, where=hist_x >= 0, color='gray', alpha=0.5)
    # plt.xlabel('X-coordinate')
    # plt.ylabel('Frequency')
    # plt.title('Frequency polygon of contour x-coordinates')
    hist_y, bins_y = np.histogram(y,density=True)
    bin_centers_Y = (bins_y[1:] + bins_y[:-1]) / 2
    # figure.add_subplot(1,2,2)
    # plt.plot(bin_centers_Y, hist_y, '-', color='black')
    # plt.fill_between(bin_centers_Y, hist_y, where=hist_y >= 0, color='gray', alpha=0.5)
    # plt.xlabel('Y-coordinate')
    # plt.ylabel('Frequency')
    # plt.title('Frequency polygon of contour y-coordinates')
    # plt.show(figure)
    
    XCheck = 0
    YCheck = 0

    xToFind = [i for i in range(0,int(img.shape[1]/3))]
    yOutput = np.interp(xToFind, bin_centers_X, hist_x)
    yOutputLessContours = 0
    for i in yOutput:
        if i < 0.0005:
            yOutputLessContours += 1
    if yOutputLessContours > 0.5*len(yOutput):
        pass
        # print("The frequency distribution is less than 0.0005 for more than 50% of the image.")
    else:
        XCheck += 1
        # print("The frequency distribution is more than 0.0005 for more than 50% of the image.")

    yToFind = [i for i in range(0,int(img.shape[0]/3))]
    xOutput = np.interp(yToFind, bin_centers_Y, hist_y)
    xOutputLessContours = 0
    for i in xOutput:
        if i < 0.0005:
            xOutputLessContours += 1
    if xOutputLessContours > 0.5*len(xOutput):
        pass
        # print("The frequency distribution is less than 0.0005 for more than 50% of the image.")
    else:
        YCheck += 1
        # print("The frequency distribution is more than 0.0005 for more than 50% of the image.")            

    
    if XCheck == 1 or YCheck == 1:
        print("The surface is Rough.")
    else:
        print("The surface is Smooth.")    

# plotDistributionCurve('/home/jerry/Desktop/SurfaceDetection/asliGround.jpg')

# def FindSurfaceType(ImgPath):
#     img = cv2.imread(ImgPath)
#     resized_img = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))
#     hist_x,bin_centers = plotDistributionCurve(ImgPath)
#     x = 200
#     y = np.interp(x, bin_centers, hist_x)
#     print(f"The estimated frequency at x={x} is {y:.2f}")
    # freq_threshold = 0.0005
    # num_pixels = resized_img.shape[0] * resized_img.shape[1]
    # num_low_freq_pixels = np.sum(y < freq_threshold) * num_pixels

    # if num_low_freq_pixels / num_pixels > 0.5: 
    #     print("The frequency distribution is less than 0.0005 for more than 50% of the image.")
    # else:
    #     print("The frequency distribution is not less than 0.0005 for more than 50% of the image.")

# FindSurfaceType('/home/jerry/Desktop/SurfaceDetection/mars1.jpg')

plotDistributionCurve('/home/jerry/Desktop/SurfaceDetection/feed970.jpg')

# currentframe = 0
# cam = cv2.VideoCapture(2)
# while True:
#     ret, frame = cam.read()
#     name = '/home/jerry/Desktop/SurfaceDetection/feed' + str(currentframe) + '.jpg'
#     print ('Creating...' + name)
#     cv2.imwrite(name, frame)
#     currentframe += 1
#     plotDistributionCurve(name)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     os.remove(name)
# cam.release()
# cv2.destroyAllWindows()