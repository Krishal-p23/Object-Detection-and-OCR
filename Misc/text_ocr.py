import easyocr
import cv2

def extract_text_from_image(image_path: str) -> str:
    ocr_reader = easyocr.Reader(lang_list=['en'], gpu=False)
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    result = ocr_reader.readtext(img)
    
    text = []
    for detection in result:
        text.append(detection[1])
        
    return ' '.join(text)        

if __name__ == "__main__":
    path = input("Enter the image path: ")
    text = extract_text_from_image(image_path=path)
    print(text)