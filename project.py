import numpy as np
import cv2
from tensorflow import keras

windowName = 'Draw an number'
img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow(windowName)
# true if mouse is pressed
drawing = False
# if True, draw rectangle. Press 'm' to toggle to curve
mode = True
(ix, iy) = (-1, -1)
# mouse callback function

# brush function


def draw_shape(event, x, y, flags, param):
    global ix, iy, drawing, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        (ix, iy) = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.circle(img, (x, y), 15, (0, 0, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.circle(img, (x, y), 15, (0, 0, 255), -1)


cv2.setMouseCallback(windowName, draw_shape)


def main():
    global mode

    while(True):
        cv2.imshow(windowName, img)

        k = cv2.waitKey(1)
        if k == ord('m') or k == ord('M'):
            mode = not mode
        # 's' is used to save the image you have drawn in Window
        elif k == ord('s') or k == ord('S'):
            cv2.imwrite('num.png', img)
        elif k == 27:
            break

    cv2.destroyAllWindows()
    img2 = cv2.imread('num.png')
    return img2


if __name__ == "__main__":
    img = main()
    img2 = img
    # converting Color of the image to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize the image
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # convert img to float32 type
    img = img.astype('float32')
    img /= 255
    img = np.reshape(img, (1, 28, 28, 1))

    # loading the previously saved model
    model1 = keras.models.load_model('model.h5')
    # predicting the number in the image
    num = model1.predict_classes(img)[0]
    # prints the number predicted
    print('number = ', num)
    f = np.zeros((500, 500, 3), dtype='uint8')
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (20, 20)
    fontScale = 1
    color = (255, 255, 0)
    thickness = 2
    print(num)
    # shows the predicted number in the image
    img2 = cv2.putText(img2, f'Predicted number = {num}', org, font,
                       fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('Predicted', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
