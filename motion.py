import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)


frame = ""
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

rval, frame = vc.read()
gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (25, 25), 0)
cv2.imshow('preview', gray1)

while rval:
    # Read frame
    rval, frame1 = vc.read()

    # Add filter
    gray2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (25, 25), 0)

    deltaframe=cv2.absdiff(gray2,gray1)
    cv2.imshow('delta',deltaframe)

    threshold = cv2.threshold(deltaframe, 40, 255, cv2.THRESH_BINARY)[1]
    threshold = cv2.dilate(threshold,None)
    cv2.imshow('threshold',threshold)

    countour, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour_size = 0
    biggest_contour = None
    for i in countour:
        if cv2.contourArea(i) > biggest_contour_size:
            biggest_contour_size = cv2.contourArea(i)
            biggest_contour = i 

    (x, y, w, h) = cv2.boundingRect(biggest_contour)
    cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('window',frame1)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()
