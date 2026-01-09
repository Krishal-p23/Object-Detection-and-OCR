from paddleocr import PaddleOCR
from PIL import Image

def resize_image(image_path, max_size=4000):
    img = Image.open(image_path)
    if max(img.size) > max_size:
        print("Pre-resizing image...")
        img.thumbnail((max_size, max_size))
        
        idx = image_path.find('.')
        resized_path = image_path[:idx] + "_resized" + image_path[idx:]
        
        img.save(resized_path)
        return resized_path
    return image_path

def extract_text_from_image(image_path: str) -> str:
    
    ocr = PaddleOCR(use_textline_orientation=False, lang='en')
    
    try:
        print("\nGathering results...\n")
        results = ocr.predict(image_path)[0]
        print("\nResults gathered...\n")
        text = ' '.join(results['rec_texts'])
        return text if text.strip() else "[No text detected]"
    
    except Exception as e:
        return f"[OCR error: {e}]"
    
if __name__ == '__main__':
    path = input("Enter image path: ").strip().strip('"')
    path = resize_image(path)
    output = extract_text_from_image(path)
    print(f"\nRecognized Text:\n{output}")
