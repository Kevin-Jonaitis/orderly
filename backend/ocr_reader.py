#!/usr/bin/env python3
"""
OCR Reader Script
Supports EasyOCR and PaddleOCR to read text from images in the menus directory.
Now includes LLM processing for menu item extraction.
"""

import os
import sys
from pathlib import Path
from paddleocr import PPStructureV3

# Add the backend directory to Python path for LLM import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processors.llm import LLMReasoner

# === OCR ENGINE SELECTION ===
# Set to 'easyocr' or 'paddleocr' to choose the OCR engine
OCR_ENGINE = 'paddleocr'  # or 'paddleocr'

MENUS_DIR = "menus"  # Directory containing menu images
IMAGE_FILENAME = "taco_bell_menu.jpg"  # Change this variable to read different files

# === LLM INSTRUCTIONS ===
LLM_INSTRUCTIONS = """<|user|>

You are reading in a MENU after OCR has been read on the image. 

Instructions:
- Only add things that you are sure are menu items, and that you have the price for. 
- Add the descriptions if they exist, but do not add them or make them up if they do not.
- Do not make up prices
- Do not add things that are not on the menu

Return an output like:

Menu Item: Description: Price


For example:
Taco Supreme: Ground beef, lettuce, cheddar cheese, diced tomatoes, sour cream, taco shell : $2.99
Bean Burrito: Flour tortilla, refried beans, cheddar cheese, red sauce  : $3.99
Cheesy Gordita Crunch: Flatbread, taco shell, seasoned beef, spicy ranch, lettuce, cheddar cheese : $11.99
Fries: : $1.99
Hot Sauce Packet: : $0.99


<|end|>"""

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
    """Read text from image using the selected OCR engine"""
    try:
        print(f"📖 Reading text from: {image_path}")
        
        if OCR_ENGINE == 'easyocr':
            results = reader.readtext(str(image_path))
            
            if not results:
                print("❌ No text found in the image")
                return None
            
            print(f"\n📋 Found {len(results)} text blocks:")
            print("=" * 50)
            
            full_text = []
            for i, (bbox, text, confidence) in enumerate(results, 1):
                print(f"{i}. Text: '{text}'")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Bounding Box: {bbox}")
                print("-" * 30)
                full_text.append(text)
            
            print("\n📄 Complete Menu Text:")
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
                return None
            
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

def process_menu_with_llm(ocr_text, llm_reasoner):
    """Process OCR text with LLM to extract menu items"""
    print("\n🧠 Processing menu with LLM...")
    print("=" * 50)
    
    # Join OCR text into a single string
    user_text = "\n".join(ocr_text) if isinstance(ocr_text, list) else str(ocr_text)
    
    # Construct the full prompt
    full_prompt = f"{LLM_INSTRUCTIONS}\n\n User said: {user_text}\n\n<|end|>\n<|assistant|>"
    
    # Print the exact prompt being sent to the LLM
    print("\n" + "="*80)
    print("🧠 EXACT PROMPT BEING SENT TO LLM (OCR Processing):")
    print("="*80)
    print(full_prompt)
    print("="*80)
    print("🧠 END OF PROMPT")
    print("="*80 + "\n")
    
    print(f"📝 Sending to LLM: {user_text[:200]}{'...' if len(user_text) > 200 else ''}")
    print("=" * 50)
    
    # Stream the LLM response
    accumulated_response = ""
    for token in llm_reasoner.generate_response_stream(full_prompt):
        print(token, end='', flush=True)
        accumulated_response += token
    
    print("\n" + "=" * 50)
    print("✅ LLM processing complete")
    
    return accumulated_response

def main():
    print(f"🔍 OCR Menu Reader with LLM Processing (engine: {OCR_ENGINE})")
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
    
    # Initialize OCR
    reader = setup_ocr()
    text_results = read_image_text(reader, image_path)
    
    if not text_results:
        print(f"\n❌ Failed to read text from {IMAGE_FILENAME}")
        return
    
    print(f"\n✅ Successfully read {len(text_results)} text blocks from {IMAGE_FILENAME}")
    
    # Initialize LLM for menu processing
    print("\n🧠 Initializing LLM for menu processing...")
    try:
        llm_reasoner = LLMReasoner()
        print("✅ LLM initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        return
    
    # Process the OCR text with LLM
    menu_items = process_menu_with_llm(text_results, llm_reasoner)
    
    print(f"\n🎯 Final extracted menu items:")
    print("=" * 50)
    print(menu_items)

if __name__ == "__main__":
    main() 