#!/usr/bin/env python3
"""
Parse OpenAI menu analysis results into separate files
"""

import subprocess
import sys
from pathlib import Path

def run_menu_analyzer():
    """Run the OpenAI menu analyzer and capture output"""
    print("🚀 Running OpenAI menu analyzer...")
    
    try:
        # Run the analyzer and capture output
        result = subprocess.run(
            [sys.executable, "openai_menu_analyzer.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        if result.returncode != 0:
            print(f"❌ Menu analyzer failed: {result.stderr}")
            return None
        
        return result.stdout
        
    except Exception as e:
        print(f"❌ Error running menu analyzer: {e}")
        return None

def parse_menu_results(output_text):
    """Parse the menu analysis output into items and prices"""
    print("📋 Parsing menu results...")
    
    # Find the section with extracted menu items
    lines = output_text.split('\n')
    menu_items = []
    
    # Look for the "Extracted Menu Items" section
    in_menu_section = False
    for line in lines:
        if "Extracted Menu Items:" in line:
            in_menu_section = True
            continue
        elif "Analysis complete!" in line:
            break
        elif in_menu_section and line.strip() and not line.startswith('='):
            menu_items.append(line.strip())
    
    return menu_items

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

def write_output_files(menu_items):
    """Write menu items to separate files"""
    print("💾 Writing output files...")
    
    items_descriptions = []
    items_prices = []
    
    for line in menu_items:
        item, description, price = parse_item_line(line)
        
        if item:  # Only add if we have at least an item name
            items_descriptions.append(f"{item}: {description}")
            items_prices.append(f"{item}: {price}")
    
    # Write items and descriptions
    with open("menu_items_descriptions.txt", "w", encoding="utf-8") as f:
        for line in items_descriptions:
            f.write(line + "\n")
    
    # Write items and prices
    with open("menu_items_prices.txt", "w", encoding="utf-8") as f:
        for line in items_prices:
            f.write(line + "\n")
    
    print(f"✅ Wrote {len(items_descriptions)} items to menu_items_descriptions.txt")
    print(f"✅ Wrote {len(items_prices)} items to menu_items_prices.txt")
    
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

def main():
    print("🍽️  Menu Results Parser")
    print("=" * 50)
    
    # Run the menu analyzer
    output = run_menu_analyzer()
    if not output:
        print("❌ Failed to get menu analysis results")
        return
    
    # Parse the results
    menu_items = parse_menu_results(output)
    if not menu_items:
        print("❌ No menu items found in output")
        return
    
    print(f"📋 Found {len(menu_items)} menu items")
    
    # Write to files
    write_output_files(menu_items)
    
    print("\n✅ Done! Check the generated files:")
    print("  - menu_items_descriptions.txt")
    print("  - menu_items_prices.txt")

if __name__ == "__main__":
    main() 