import cv2
import numpy as np
import utlis
######################################
path="2.jpg"
widthImg=700
heightImg=700
questions=5
choices=5
ans=[1,3,0,2,4]
webcamFeed=True
cameraNo=0
#################################
cap=cv2.VideoCapture(cameraNo)
cap.set(10,150)

while True:
    if webcamFeed:success,img=cap.read()
    else:img=cv2.imread(path)

    img=cv2.imread(path)

    #preprocessing
    img=cv2.resize(img,(widthImg,heightImg))
    imgContours=img.copy()
    imgFinal=img.copy()
    imgBiggestContours=img.copy()


    imgGray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny=cv2.Canny(imgBlur,10,50)
    try:
        contours, hierarchy=cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours,contours,-1,(0,255,0),10)

        #findrect
        rectCon=utlis.rectContour(contours)
        biggestContour=utlis.getCornerPoints(rectCon[0])
        gradePoints=utlis.getCornerPoints(rectCon[1])
        #print((biggestContour))

        if biggestContour.size !=0 and gradePoints.size !=0:
            cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),10)
            cv2.drawContours(imgBiggestContours,gradePoints,-1,(255,0,0),20)

            biggestContour= utlis.reorder(biggestContour)
            gradePoints=utlis.reorder(gradePoints)

            pts1=np.float32(biggestContour)
            pts2=np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
            matrix =cv2.getPerspectiveTransform(pts1,pts2)
            imgWarpColored=cv2.warpPerspective(img,matrix,(widthImg,heightImg))

            ptsG1=np.float32(gradePoints)
            ptsG2=np.float32([[0,0],[325,0],[0,150],[325,150]])
            matrixG =cv2.getPerspectiveTransform(ptsG1,ptsG2)
            imgGradeDisplay=cv2.warpPerspective(img,matrixG,(325,150))
            #cv2.imshow("grade",imgGradeDisplay)

            #apply threshold
            imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
            imgThresh =cv2.threshold(imgWarpGray,170,255,cv2.THRESH_BINARY_INV)[1]

            boxex=utlis.splitBoxes(imgThresh)
            #cv2.imshow("Thresh",boxex[2])

            myPixelVal=np.zeros((questions,choices))
            countC=0
            countR=0

            for image in boxex:
                totalPixels=cv2.countNonZero(image)
                myPixelVal[countR][countC]=totalPixels
                countC += 1
                if (countC == choices): countC = 0;countR += 1
                #print((myPixelVal))

            myIndex=[]
            for x in range(0,questions):
                arr=myPixelVal[x]
                myIndexVal=np.where(arr==np.amax(arr))
                myIndex.append(myIndexVal[0][0])
            #print(myIndex)

            #grading
            grading=[]
            for x in range(0,questions):
                if ans[x]==myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)
            #print((grading))

            score=(sum(grading)/questions)*100
            print(score)

            #displaying ans
            imgResult=imgWarpColored.copy()
            imgResult=utlis.showAnswers(imgResult,myIndex,grading, ans, questions, choices)
            imRawDrawing=np.zeros_like(imgWarpColored)
            imRawDrawing=utlis.showAnswers(imRawDrawing,myIndex,grading, ans, questions, choices)
            inMatrix = cv2.getPerspectiveTransform(pts2, pts1)
            imgInvWarp = cv2.warpPerspective(imRawDrawing, inMatrix, (widthImg, heightImg))



            imgRawGrade=np.zeros_like(imgGradeDisplay)
            cv2.putText(imgRawGrade,str(int(score))+"%",(50,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3)
            invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1)
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))



            imgFinal=cv2.addWeighted(imgFinal,1,imgInvWarp,1,0)
            imgFinal=cv2.addWeighted(imgFinal,1,imgInvGradeDisplay,1,0)





        imgBlank=np.zeros_like(img)
        imageArray = [[img,imgGray,imgBlur,imgCanny],
                      [imgContours,imgBiggestContours,imgWarpColored,imgThresh],
                      [imgResult,imRawDrawing,imgInvWarp,imgFinal]]
    except:
        imgBlank = np.zeros_like(img)
        imageArray = [[img, imgGray, imgBlur, imgCanny],
                      [imgBlank, imgBlank, imgBlank, imgBlank],
                      [imgBlank, imgBlank, imgBlank, imgBlank]]

    lables=[["original","gray","blur","canny"],
            ["contours","biggest contours","warp","threshold"],
            ["result","raw drawing","inverse warp","final"]]
    imgStacked=utlis.stackImages(imageArray,0.3)

    cv2.imshow("final",imgFinal)
    cv2.imshow("stacked images",imgStacked)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("final result.jpg",imgFinal)
        cv2.waitKey(300)