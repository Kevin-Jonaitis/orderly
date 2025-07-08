#!/usr/bin/env python3
"""
OCR Reader Script
Supports EasyOCR and PaddleOCR to read text from images in the menus directory.
"""

import os
import sys
from pathlib import Path
from paddleocr import PPStructureV3

# === OCR ENGINE SELECTION ===
# Set to 'easyocr' or 'paddleocr' to choose the OCR engine
OCR_ENGINE = 'paddleocr'  # or 'paddleocr'

MENUS_DIR = "menus"  # Directory containing menu images
IMAGE_FILENAME = "in_n_out_menu.jpg"  # Change this variable to read different files

# --- OCR Engine Imports ---
if OCR_ENGINE == 'easyocr':
    import easyocr
elif OCR_ENGINE == 'paddleocr':
    from paddleocr import PaddleOCR
else:
    raise ValueError(f"Unknown OCR_ENGINE: {OCR_ENGINE}")

def setup_ocr():
    """Initialize the selected OCR engine"""
    print(f"🔍 Initializing {OCR_ENGINE}...")
    if OCR_ENGINE == 'easyocr':
        reader = easyocr.Reader(['en'])
    elif OCR_ENGINE == 'paddleocr':
        # reader = PPStructureV3(use_doc_orientation_classify=False,
        #               use_doc_unwarping=False)
        reader = PaddleOCR(use_textline_orientation=True,
                          lang='en')
    print(f"✅ {OCR_ENGINE} initialized successfully")
    return reader

def read_image_text(reader, image_path):
    """Read text from an image using the selected OCR engine"""
    print(f"📖 Reading text from: {image_path}")
    try:
        if OCR_ENGINE == 'easyocr':
            results = reader.readtext(str(image_path))
            if not results:
                print("❌ No text found in the image")
                return
            print(f"\n📋 Found {len(results)} text blocks:")
            print("=" * 50)
            full_text = []
            for i, (bbox, text, confidence) in enumerate(results, 1):
                print(f"{i}. Text: '{text}'")
                print(f"   Confidence: {confidence:.2f}")
                print(f"   Bounding Box: {bbox}")
                print("-" * 30)
                full_text.append(text)
            print("\n📄 Complete Text:")
            print("=" * 50)
            print("\n".join(full_text))
            return full_text
        elif OCR_ENGINE == 'paddleocr':
            # results = reader.predict('menus/' + IMAGE_FILENAME, return_ocr_result_in_table=False)
            results = reader.predict('menus/' + IMAGE_FILENAME)

            for res in results:
                print(res)

            if not results or not results[0]:
                print("❌ No text found in the image")
                return
            
            # Extract the result data from the OCRResult object
            result_data = results[0]
            rec_texts = result_data.get('rec_texts', [])
            rec_scores = result_data.get('rec_scores', [])
            rec_boxes = result_data.get('rec_boxes', [])
            
            print(f"\n📋 Found {len(rec_texts)} text blocks:")
            print("=" * 50)
            
            full_text = []
            for i, (text, score) in enumerate(zip(rec_texts, rec_scores), 1):
                print(f"{i}. Text: '{text}'")
                print(f"   Confidence: {score:.3f}")
                if i <= len(rec_boxes):
                    print(f"   Bounding Box: {rec_boxes[i-1].tolist()}")
                print("-" * 30)
                full_text.append(text)
            
            print("\n📄 Complete Menu Text:")
            print("=" * 50)
            print("\n".join(full_text))
            
            # Print additional arrays for debugging
            print(f"\n🔍 Additional Data:")
            print(f"   Recognition Scores: {len(rec_scores)} scores")
            print(f"   Recognition Boxes: {len(rec_boxes)} boxes")
            
            return full_text
    except Exception as e:
        print(f"❌ Error reading image: {e}")
        return None

def main():
    print(f"🔍 OCR Menu Reader (engine: {OCR_ENGINE})")
    print("=" * 50)
    backend_dir = Path(__file__).parent
    menus_dir = backend_dir / MENUS_DIR
    image_path = menus_dir / IMAGE_FILENAME
    if not menus_dir.exists():
        print(f"❌ Menus directory not found: {menus_dir}")
        print("Please create the 'menus' directory in the backend folder")
        return
    if not image_path.exists():
        print(f"❌ Image file not found: {image_path}")
        print(f"Available files in {menus_dir}:")
        for file in menus_dir.glob("*"):
            if file.is_file():
                print(f"  - {file.name}")
        return
    reader = setup_ocr()
    text_results = read_image_text(reader, image_path)
    if text_results:
        print(f"\n✅ Successfully read {len(text_results)} text blocks from {IMAGE_FILENAME}")
    else:
        print(f"\n❌ Failed to read text from {IMAGE_FILENAME}")

if __name__ == "__main__":
    main() 