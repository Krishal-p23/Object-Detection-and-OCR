from ultralytics import YOLO
import text_ocr

model_name = 'yolo_fine_tuned.pt'

def classifier(image_path: str) -> str:
    yolo = YOLO(model_name)
    results = yolo.predict(image_path, conf=0.25)
    
    object_names = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes.data.tolist():
            cls_id = int(box[5])
            object_names.append(results[0].names[cls_id])
     
    object_names = set(object_names)
    if 'text' in object_names:    
        text = text_ocr.extract_text_from_image(image_path)
        object_names.discard('text')
    
    return f"{' '.join(object_names)} {text}"

if __name__ == "__main__":
    path = input("Enter the image path: ")
    text = classifier(image_path=path)
    print(text)