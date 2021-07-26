#---------------
#
# Выполнить преобразование перспективы четырехугольного листа.
#
#---------------

from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt


factorDraw = 0.5        # Фактор уменьшения начального изображения. Используется для отладки.
matrixBlure = (15, 15)  # Параметры матрицы для размытия изображения.
numberTreshold = 160    # Пороговое значение для получения бинарного изображения

harrisParam1 = 9        # Параметры алгоритма Харриса.
harrisParam2 = 31
harrisParam3 = 0.04


#---------------
# Отрисовать изображение. Используется для отладки....
#---------------
def CheckImage( img ):
    winname = 'SomeName'
    height, width = img.shape[:2]

    imgDraw = cv2.resize(img, (int(width * factorDraw), int(height * factorDraw)), cv2.INTER_LINEAR)

    cv2.imshow(winname, imgDraw)
    cv2.waitKey(0)


#---------------
# Поиск контура изображения.
# Вернуть изображение замкнутого контура на белом листе.
#---------------
def ContourImage( image ):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, matrixBlure, 0)

    T, thresh_img = cv2.threshold(blurred, numberTreshold, 255, cv2.THRESH_BINARY)

    #CheckImage( thresh_img )

    cnts, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    arrayPoints = cnts[0] # Набор точек составляющих контур рамки.

    height, width = image.shape[:2]
    imgWhite = np.ones((height,width,3) , np.uint8) *255

    for i in range(0, len(arrayPoints)-1):
        point0 = arrayPoints[i][0]
        point1 = arrayPoints[i+1][0]
        cv2.line(imgWhite, point0, point1, (0, 0, 255), 4)
    cv2.line(imgWhite, arrayPoints[len(arrayPoints)-1][0], arrayPoints[0][0], (0, 0, 255), 4)
    imgWhite = cv2.cvtColor(imgWhite, cv2.COLOR_RGB2GRAY)

    return imgWhite


#---------------
# Получить набор углов контура. Используется алгоритм Харриса.
#---------------
def CornerContour( image ):

    dst = cv2.cornerHarris(image, harrisParam1, harrisParam2, harrisParam3)
    dst = cv2.dilate(dst,None)
  
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(image,np.float32(centroids),(5,5),(-1,-1),criteria) #Уточнение углов

    corners = np.delete( corners, 0, 0 ) # Удалить центр рамки
    return corners


#---------------
# Трансформировать изображение в новые угловые точки( конвертировать перспективу )
#---------------
def PerspectiveWarp( initImage, newCorners ):

    height, width = initImage.shape[:2]
    dst = np.float32([(0, 0),
                      (width, 0),
                      (0, height),
                      (width, height)])

    M = cv2.getPerspectiveTransform(newCorners, dst)
    warped = cv2.warpPerspective(initImage, M, (width, height), flags=cv2.INTER_LINEAR)

    return warped



def ConvertPicture( namePicture ):

    initImage = cv2.imread( namePicture, cv2.IMREAD_COLOR ) # Прочитать изображение

    if initImage is not None :
        #CheckImage( initImage )
        
        contourImage = ContourImage( initImage ) # Поиск контура. Вернуть изображение замкнутого контура на белом листе
        
        #CheckImage( contourImage )
        
        pointCorners = CornerContour( contourImage ) # Получить набор углов контура
        
        #for i in range(0, len(pointCorners)):
        #    contourImage = cv2.circle(contourImage, (int(pointCorners[i][0]), int(pointCorners[i][1])), radius=10, color=(0, 0, 255), thickness=1)
        #CheckImage( contourImage )
        
        warped = PerspectiveWarp( initImage, pointCorners ) # Трансформировать изображение в новые угловые точки
        
        #CheckImage( warped )
        
        cv2.imwrite( 'warped.jpg' , warped) # Сохранить новое изображение

        print ( 'Saved' )
    else:
        print ( 'File not found' )

   
print ( 'Enter file name \n' )
fileName = input()

ConvertPicture( fileName )
