#!/usr/bin/env python3
"""
Script to fix duplicate IDs in scenario files and scenario_ids in result files
Makes all IDs unique by appending suffixes when duplicates are found
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def fix_scenario_file_ids(file_path):
    """Fix duplicate IDs in a scenario file"""
    print(f"Checking scenario file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'scenarios' not in data:
        print(f"  No 'scenarios' key found in {file_path}")
        return False
    
    # Track IDs and their occurrences
    id_counts = defaultdict(int)
    scenarios = data['scenarios']
    changes_made = False
    
    # First pass: count occurrences
    for scenario in scenarios:
        if 'id' in scenario:
            id_counts[scenario['id']] += 1
    
    # Second pass: fix duplicates
    id_usage = defaultdict(int)
    for i, scenario in enumerate(scenarios):
        if 'id' not in scenario:
            continue
            
        original_id = scenario['id']
        id_usage[original_id] += 1
        
        # If this ID appears multiple times and this is not the first occurrence
        if id_counts[original_id] > 1 and id_usage[original_id] > 1:
            new_id = f"{original_id}_{id_usage[original_id] - 1}"
            print(f"  Changing duplicate ID '{original_id}' to '{new_id}'")
            scenario['id'] = new_id
            changes_made = True
    
    # Write back if changes were made
    if changes_made:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Fixed duplicates in {file_path}")
    else:
        print(f"  ✓ No duplicates found in {file_path}")
    
    return changes_made

def fix_result_file_ids(file_path):
    """Fix duplicate scenario_ids in a result file"""
    print(f"Checking result file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'results' not in data:
        print(f"  No 'results' key found in {file_path}")
        return False
    
    # Track scenario_ids and their occurrences
    id_counts = defaultdict(int)
    results = data['results']
    changes_made = False
    
    # First pass: count occurrences
    for result in results:
        if 'scenario_id' in result:
            id_counts[result['scenario_id']] += 1
    
    # Second pass: fix duplicates
    id_usage = defaultdict(int)
    for i, result in enumerate(results):
        if 'scenario_id' not in result:
            continue
            
        original_id = result['scenario_id']
        id_usage[original_id] += 1
        
        # If this scenario_id appears multiple times and this is not the first occurrence
        if id_counts[original_id] > 1 and id_usage[original_id] > 1:
            new_id = f"{original_id}_{id_usage[original_id] - 1}"
            print(f"  Changing duplicate scenario_id '{original_id}' to '{new_id}'")
            result['scenario_id'] = new_id
            changes_made = True
    
    # Write back if changes were made
    if changes_made:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Fixed duplicates in {file_path}")
    else:
        print(f"  ✓ No duplicates found in {file_path}")
    
    return changes_made

def find_duplicates_in_file(file_path, id_key):
    """Find and report duplicates in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        collection_key = 'scenarios' if id_key == 'id' else 'results'
        if collection_key not in data:
            return []
        
        ids = []
        for item in data[collection_key]:
            if id_key in item:
                ids.append(item[id_key])
        
        # Find duplicates
        id_counts = defaultdict(int)
        for id_val in ids:
            id_counts[id_val] += 1
        
        duplicates = [id_val for id_val, count in id_counts.items() if count > 1]
        return duplicates
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def main():
    print("=== Duplicate ID Fixer ===\n")
    
    # Find scenario files
    scenario_dir = Path("scenarios")
    result_dir = Path("results")
    
    total_changes = 0
    
    # Process scenario files
    if scenario_dir.exists():
        print("1. Checking scenario files for duplicate IDs...")
        scenario_files = list(scenario_dir.glob("*.json"))
        
        for file_path in scenario_files:
            # First, report any duplicates found
            duplicates = find_duplicates_in_file(file_path, 'id')
            if duplicates:
                print(f"  Found duplicates in {file_path}: {duplicates}")
            
            # Fix the duplicates
            if fix_scenario_file_ids(file_path):
                total_changes += 1
        
        print(f"\nProcessed {len(scenario_files)} scenario files\n")
    else:
        print("No scenarios directory found\n")
    
    # Process result files
    if result_dir.exists():
        print("2. Checking result files for duplicate scenario_ids...")
        result_files = list(result_dir.glob("*.json"))
        
        for file_path in result_files:
            # First, report any duplicates found
            duplicates = find_duplicates_in_file(file_path, 'scenario_id')
            if duplicates:
                print(f"  Found duplicates in {file_path}: {duplicates}")
            
            # Fix the duplicates
            if fix_result_file_ids(file_path):
                total_changes += 1
        
        print(f"\nProcessed {len(result_files)} result files\n")
    else:
        print("No results directory found\n")
    
    print(f"=== Summary ===")
    print(f"Total files with changes: {total_changes}")
    
    if total_changes > 0:
        print("✓ All duplicate IDs have been fixed!")
    else:
        print("✓ No duplicate IDs found!")

if __name__ == "__main__":
    main()
