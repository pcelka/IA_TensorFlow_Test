import cv2
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")
img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "Snickers_degrade_{}.png".format(img_counter)
        # Cropping an image Y1:Y2 , X1:X2
        cropped_image = frame[115:365, 140:390]
        cv2.imwrite(img_name, cropped_image)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
