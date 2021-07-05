import pytesseract
import pytesseract
import cv2

def img_to_text():
    image = cv2.imread("test1.jpg")

    # Add custom option
    custom_config = r'--oem 3 --psm 6'
    print(pytesseract.image_to_string(image, config=custom_config))
