#!/usr/bin/env python3
"""
Script to fix malformed raw_input parameters in ParsedCommand constructor calls.

This fixes cases where raw_input got duplicated or malformed during the previous fix.
"""

import os
import re
from pathlib import Path


def fix_malformed_raw_input_in_file(file_path: Path) -> bool:
    """Fix malformed raw_input parameters in ParsedCommand constructor calls.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        True if any replacements were made, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = False
        
        # Fix malformed raw_input with duplications like:
        # raw_input="/command(..., raw_input="/command(..., raw_input="/command(...)")")")
        pattern1 = r'raw_input="(/[^"]+)(?:, raw_input="[^"]+)*"\)+"'
        replacement1 = r'raw_input="\1"'
        content, count1 = re.subn(pattern1, replacement1, content)
        
        if count1 > 0:
            print(f"  - Fixed {count1} malformed raw_input duplications")
            changes_made = True
        
        # Fix cases where raw_input has nested quotes or f-strings gone wrong
        pattern2 = r'raw_input=f"\{([^}]+)\}\(\.\.\.(?:, raw_input="[^"]+)*"\)+"'
        replacement2 = r'raw_input=f"/{{\1}}(...)"'
        content, count2 = re.subn(pattern2, replacement2, content)
        
        if count2 > 0:
            print(f"  - Fixed {count2} malformed f-string raw_input")
            changes_made = True
        
        # Fix any remaining malformed patterns
        pattern3 = r'raw_input="([^"]+)"\)+"'
        replacement3 = r'raw_input="\1"'
        content, count3 = re.subn(pattern3, replacement3, content)
        
        if count3 > 0:
            print(f"  - Fixed {count3} extra closing parentheses")
            changes_made = True
        
        if changes_made:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"  - Error processing {file_path}: {e}")
        return False


def main():
    """Main function to scan and fix test files."""
    print("🔍 Scanning for malformed raw_input parameters...")
    
    # Define test directories to scan
    test_dirs = [
        "tests/unit/commands/handlers/",
        "tests/integration/chat/",
        "tests/unit/commands/",
    ]
    
    total_files_processed = 0
    total_files_changed = 0
    
    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if not test_path.exists():
            print(f"⚠️  Directory {test_dir} not found, skipping...")
            continue
            
        print(f"\n📁 Processing {test_dir}...")
        
        # Find all Python files recursively
        python_files = list(test_path.rglob("*.py"))
        
        for py_file in python_files:
            # Skip files that don't contain ParsedCommand
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                if 'raw_input=' not in file_content or ', raw_input=' not in file_content:
                    continue
            except Exception:
                continue
            
            total_files_processed += 1
            print(f"  📄 {py_file}")
            
            if fix_malformed_raw_input_in_file(py_file):
                total_files_changed += 1
                print(f"    ✅ Changes made")
            else:
                print(f"    ➡️  No changes needed")
    
    print(f"\n📊 Summary:")
    print(f"  - Files processed: {total_files_processed}")
    print(f"  - Files changed: {total_files_changed}")
    
    if total_files_changed > 0:
        print(f"\n✅ Successfully fixed malformed raw_input parameters!")
    else:
        print(f"\n ℹ️  No malformed raw_input parameters found that needed fixing.")


if __name__ == "__main__":
    main()