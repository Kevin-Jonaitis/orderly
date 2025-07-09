#!/usr/bin/env python3
"""
OCR and Parse Menu - Combined OpenAI GPT-4 Vision Menu Analyzer and Parser
Uploads menu images to OpenAI, extracts menu items and prices, and parses them into structured output.
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
    """Analyze menu image using GPT-4o"""
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
Do not include numbers in front of the item name.

For example:
Taco Supreme: Ground beef, lettuce, cheddar cheese, diced tomatoes, sour cream, taco shell :$2.99
Bean Burrito: Flour tortilla, refried beans, cheddar cheese, red sauce :$3.99
Combo #1: 2 Taco Supremes, 1 Pepsi(small) :$5.99
Pepsi(small): :$1.99
Pepsi(large): :$1.99
Fries: :$1.99

"""
    
    try:
        print("🚀 Sending to OpenAI for analysis...")
        
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

def parse_item_line(line):
    """Parse a single menu item line in format: item:description:price"""
    parts = line.split(':')
    
    if len(parts) >= 3:
        item = parts[0].strip()
        description = parts[1].strip()
        price = parts[2].strip()
        return item, description, price
    elif len(parts) == 2:
        # Handle case where there's no description
        item = parts[0].strip()
        price = parts[1].strip()
        return item, "", price
    else:
        # Handle malformed lines
        return line.strip(), "", ""

def parse_menu_results(analysis_output):
    """Parse the menu analysis output into items and prices"""
    print("📋 Parsing menu results...")
    
    # Split the output into lines and filter out empty lines
    lines = [line.strip() for line in analysis_output.split('\n') if line.strip()]
    
    menu_items = []
    for line in lines:
        # Skip lines that are just separators or headers
        if line.startswith('=') or 'Extracted Menu Items:' in line or 'Analysis complete!' in line:
            continue
        menu_items.append(line)
    
    return menu_items

def write_output_files(menu_items, menus_dir):
    """Write menu items to separate files in the menus directory"""
    print("💾 Writing output files...")
    
    items_descriptions = []
    items_prices = []
    
    for line in menu_items:
        item, description, price = parse_item_line(line)
        
        if item:  # Only add if we have at least an item name
            items_descriptions.append(f"{item}: {description}")
            items_prices.append(f"{item}: {price}")
    
    # Write items and descriptions
    descriptions_file = menus_dir / "menu_items_descriptions.txt"
    with open(descriptions_file, "w", encoding="utf-8") as f:
        for line in items_descriptions:
            f.write(line + "\n")
    
    # Write items and prices
    prices_file = menus_dir / "menu_items_prices.txt"
    with open(prices_file, "w", encoding="utf-8") as f:
        for line in items_prices:
            f.write(line + "\n")
    
    print(f"✅ Wrote {len(items_descriptions)} items to {descriptions_file}")
    print(f"✅ Wrote {len(items_prices)} items to {prices_file}")
    
    # Show preview
    print("\n📋 Preview of items and descriptions:")
    print("=" * 40)
    for line in items_descriptions[:5]:  # Show first 5
        print(line)
    if len(items_descriptions) > 5:
        print(f"... and {len(items_descriptions) - 5} more")
    
    print("\n💰 Preview of items and prices:")
    print("=" * 40)
    for line in items_prices[:5]:  # Show first 5
        print(line)
    if len(items_prices) > 5:
        print(f"... and {len(items_prices) - 5} more")

def process_menu_image(image_path):
    """Process a menu image through the entire OCR pipeline"""
    print(f"🤖 Processing menu image: {image_path}")
    print("=" * 50)
    
    try:
        # Load API key
        api_key = load_api_key()
        if not api_key:
            print("❌ Failed to load API key")
            return False
        
        # Analyze the menu image
        analysis_output = analyze_menu_image(image_path, api_key)
        if not analysis_output:
            print("❌ Failed to analyze menu image")
            return False
        
        # Parse the results
        menu_items = parse_menu_results(analysis_output)
        if not menu_items:
            print("❌ No menu items found in analysis")
            return False
        
        # Set up paths
        backend_dir = Path(__file__).parent
        menus_dir = backend_dir / "menus"
        
        # Write output files
        write_output_files(menu_items, menus_dir)
        
        print("✅ Menu processing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error processing menu image: {e}")
        return False

def main():
    print("🤖 OCR and Parse Menu - Combined OpenAI GPT-4 Vision Menu Analyzer")
    print("=" * 70)
    
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
    
    # Step 1: Analyze the image with OpenAI
    # The process_menu_image function now handles the entire pipeline
    success = process_menu_image(selected_image)
    
    if not success:
        print("❌ Failed to process menu image")
        return
    
    print("\n✅ Complete! Check the generated files:")
    print(f"  - {menus_dir}/menu_items_descriptions.txt")
    print(f"  - {menus_dir}/menu_items_prices.txt")

if __name__ == "__main__":
    main() 