#!/usr/bin/env python3
"""
OpenAI GPT-4 Vision Menu Analyzer
Uploads menu images to OpenAI and extracts menu items and prices.
"""

import os
import sys
import base64
import re
from openai import OpenAI
from pathlib import Path

def load_api_key():
    """Load OpenAI API key from file with proper cleaning"""
    api_key_file = Path("open-ai-api-key.txt")
    # Read the file with UTF-8 encoding
    with open(api_key_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        return content

def encode_image(image_path):
    """Encode image to base64"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ Error encoding image {image_path}: {e}")
        return None

def analyze_menu_image(image_path, api_key):
    """Analyze menu image using GPT-4o-mini"""
    print(f"🔍 Analyzing menu image: {image_path}")
    print("=" * 50)
    
    # Encode the image
    image_base64 = encode_image(image_path)
    if not image_base64:
        return None
    
    # Set up OpenAI client with new API
    client = OpenAI(api_key=api_key)
    
    # Instructions for menu extraction
    instructions = """You are an expert menu parser. Given an image, carefully extract and format menu items as 'Item: Description: Price'. Think carefully and reason before writing output. 
Only include items that you are confident are menu items with clear prices. If you're unsure about a price or item, don't include it.
Have each individual item on a new line. Each possible combination of an item should have it's own line.
Make sure to include combos if they exist and name them with the number. 
Only incldue the price at the end of each line, nothing more.
Split out different flavors/types to their own line.

For example:
Taco Supreme: Ground beef, lettuce, cheddar cheese, diced tomatoes, sour cream, taco shell :$2.99
Bean Burrito: Flour tortilla, refried beans, cheddar cheese, red sauce :$3.99
Combo #1: 2 Taco Supremes, 1 Pepsi(small) :$5.99
Pepsi(small): :$1.99
Pepsi(large): :$1.99
Fries: :$1.99

"""
    
    try:
        print("🚀 Sending to my boi chat...")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instructions},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Extract and return the response
        content = response.choices[0].message.content
        return content
        
    except Exception as e:
        print(f"❌ Error calling OpenAI API: {e}")
        return None

def main():
    print("🤖 OpenAI GPT-4o-mini Vision Menu Analyzer")
    print("=" * 50)
    
    # Load API key
    api_key = load_api_key()
    if not api_key:
        print("❌ Failed to load API key. Please check the file encoding.")
        return
    
    # Set up paths
    backend_dir = Path(__file__).parent
    menus_dir = backend_dir / "menus"
    
    if not menus_dir.exists():
        print(f"❌ Menus directory not found: {menus_dir}")
        return
    
    # List available images
    image_files = list(menus_dir.glob("*.jpg")) + list(menus_dir.glob("*.png")) + list(menus_dir.glob("*.jpeg"))
    
    if not image_files:
        print(f"❌ No image files found in {menus_dir}")
        print("Please add some menu images to the menus directory")
        return
    
    print(f"📁 Found {len(image_files)} image files:")
    for i, img_file in enumerate(image_files, 1):
        print(f"  {i}. {img_file.name}")
    
    # For now, analyze the first image (you can modify this to select specific images)
    selected_image = image_files[0]
    print(f"\n🎯 Analyzing: {selected_image.name}")
    
    # Analyze the image
    result = analyze_menu_image(selected_image, api_key)
    
    if result:
        print("\n📋 Extracted Menu Items:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        print("✅ Analysis complete!")
    else:
        print("❌ Failed to analyze image")

if __name__ == "__main__":
    main() 