#!/usr/bin/env python3
"""
Script to fix 'generate' method calls to use 'infer' instead in test files.

This addresses the issue where tests are trying to patch 'generate' method
on BaseInferenceEngine mocks, but the actual interface uses 'infer' method.
"""

import os
import re
from pathlib import Path


def find_and_replace_generate_calls(file_path: Path) -> bool:
    """Find and replace 'generate' method calls with 'infer' in a file.
    
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
        
        # Pattern 1: patch.object(mock_engine, 'generate')
        pattern1 = r"patch\.object\(([^,]+),\s*['\"]generate['\"]"
        replacement1 = r"patch.object(\1, 'infer'"
        content, count1 = re.subn(pattern1, replacement1, content)
        if count1 > 0:
            print(f"  - Fixed {count1} patch.object(*, 'generate') calls")
            changes_made = True
        
        # Pattern 2: mock_engine.generate = ...
        pattern2 = r"([a-zA-Z_][a-zA-Z0-9_]*\.generate)\s*="
        replacement2 = r"\1".replace("generate", "infer") + " ="
        content, count2 = re.subn(pattern2, replacement2, content)
        if count2 > 0:
            print(f"  - Fixed {count2} mock_engine.generate = ... assignments")
            changes_made = True
        
        # Pattern 3: mock_generate variable names and related calls
        # First, find variable assignments like: mock_generate = ...
        pattern3a = r"(\s+)mock_generate(\s*=)"
        replacement3a = r"\1mock_infer\2"
        content, count3a = re.subn(pattern3a, replacement3a, content)
        
        # Then fix the variable references
        pattern3b = r"mock_generate\."
        replacement3b = "mock_infer."
        content, count3b = re.subn(pattern3b, replacement3b, content)
        
        # And standalone mock_generate references
        pattern3c = r"\bmock_generate\b"
        replacement3c = "mock_infer"
        content, count3c = re.subn(pattern3c, replacement3c, content)
        
        if count3a + count3b + count3c > 0:
            print(f"  - Fixed {count3a + count3b + count3c} mock_generate variable references")
            changes_made = True
        
        # Pattern 4: Comments mentioning 'generate'
        pattern4 = r"(#.*?)generate(.*)"
        def replace_comment(match):
            return match.group(1) + "infer" + match.group(2)
        content, count4 = re.subn(pattern4, replace_comment, content)
        if count4 > 0:
            print(f"  - Fixed {count4} comments mentioning 'generate'")
            changes_made = True
        
        # Pattern 5: String literals mentioning generate in test contexts
        pattern5 = r"(['\"])([^'\"]*?)generate([^'\"]*?)\1"
        def replace_string(match):
            quote = match.group(1)
            before = match.group(2)
            after = match.group(3)
            # Only replace if it looks like it's referring to the method
            if "mock" in before.lower() or "engine" in before.lower() or "method" in (before + after).lower():
                return f"{quote}{before}infer{after}{quote}"
            return match.group(0)
        content, count5 = re.subn(pattern5, replace_string, content)
        if count5 > 0:
            print(f"  - Fixed {count5} string literals mentioning 'generate'")
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
    print("🔍 Scanning for 'generate' method calls that should be 'infer'...")
    
    # Define test directories to scan
    test_dirs = [
        "tests/unit/commands/",
        "tests/integration/chat/",
        "tests/utils/"
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
            total_files_processed += 1
            print(f"  📄 {py_file}")
            
            if find_and_replace_generate_calls(py_file):
                total_files_changed += 1
                print(f"    ✅ Changes made")
            else:
                print(f"    ➡️  No changes needed")
    
    print(f"\n📊 Summary:")
    print(f"  - Files processed: {total_files_processed}")
    print(f"  - Files changed: {total_files_changed}")
    
    if total_files_changed > 0:
        print(f"\n✅ Successfully fixed 'generate' → 'infer' method calls!")
        print(f"   You should now be able to run tests without AttributeError about missing 'generate' method.")
    else:
        print(f"\n ℹ️  No 'generate' method calls found that needed fixing.")


if __name__ == "__main__":
    main()