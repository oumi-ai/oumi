#!/usr/bin/env python3
"""
Script to fix missing raw_input parameters in ParsedCommand constructor calls.

This addresses the issue where ParsedCommand objects are created without the required
raw_input parameter, causing TypeError exceptions.
"""

import os
import re
from pathlib import Path


def fix_missing_raw_input_in_file(file_path: Path) -> bool:
    """Fix missing raw_input parameters in ParsedCommand constructor calls.
    
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
        
        # Pattern 1: ParsedCommand(command="...", args=[...], kwargs={})
        # Missing raw_input parameter
        pattern1 = r'ParsedCommand\(\s*command=([^,]+),\s*args=([^,]+),\s*kwargs=([^)]+)\)'
        
        def replace_parsed_command(match):
            command = match.group(1)
            args = match.group(2)
            kwargs = match.group(3)
            
            # Extract command name to create a reasonable raw_input
            if '"' in command or "'" in command:
                # Extract quoted command name
                cmd_match = re.search(r'["\']([^"\']+)["\']', command)
                if cmd_match:
                    cmd_name = cmd_match.group(1)
                    raw_input = f'"/{cmd_name}(...)"'
                else:
                    raw_input = '"/unknown(...)"'
            else:
                # Variable reference
                raw_input = f'f"/{{{command}}}(...)"'
            
            return f'ParsedCommand(command={command}, args={args}, kwargs={kwargs}, raw_input={raw_input})'
        
        new_content, count1 = re.subn(pattern1, replace_parsed_command, content)
        
        if count1 > 0:
            print(f"  - Fixed {count1} ParsedCommand constructor calls")
            content = new_content
            changes_made = True
        
        # Pattern 2: Look for any remaining ParsedCommand calls that might be malformed
        # and don't have raw_input
        pattern2 = r'ParsedCommand\([^)]*\)'
        matches = re.findall(pattern2, content)
        remaining_issues = [m for m in matches if 'raw_input' not in m]
        
        if remaining_issues:
            print(f"  - Found {len(remaining_issues)} potentially problematic ParsedCommand calls:")
            for issue in remaining_issues[:3]:  # Show first 3
                print(f"    {issue}")
        
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
    print("🔍 Scanning for missing raw_input parameters in ParsedCommand calls...")
    
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
                if 'ParsedCommand(' not in file_content:
                    continue
            except Exception:
                continue
            
            total_files_processed += 1
            print(f"  📄 {py_file}")
            
            if fix_missing_raw_input_in_file(py_file):
                total_files_changed += 1
                print(f"    ✅ Changes made")
            else:
                print(f"    ➡️  No changes needed")
    
    print(f"\n📊 Summary:")
    print(f"  - Files processed: {total_files_processed}")
    print(f"  - Files changed: {total_files_changed}")
    
    if total_files_changed > 0:
        print(f"\n✅ Successfully fixed missing raw_input parameters!")
        print(f"   ParsedCommand constructor calls now include required raw_input parameter.")
    else:
        print(f"\n ℹ️  No missing raw_input parameters found that needed fixing.")


if __name__ == "__main__":
    main()