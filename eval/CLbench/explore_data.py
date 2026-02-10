#!/usr/bin/env python3
"""Explore CL-bench dataset to understand structure and find Procedural Task Execution samples."""
import json
from collections import Counter
from datasets import load_dataset

print("Loading CL-bench dataset...")
ds = load_dataset('tencent/CL-bench', split='train')
print(f'Total samples: {len(ds)}')
print(f'Columns: {ds.column_names}')

# Look at first sample structure
sample = ds[0]
print(f'\n--- Sample 0 Structure ---')
print(f'messages type: {type(sample["messages"])}')
print(f'messages length: {len(sample["messages"])}')
for i, msg in enumerate(sample['messages']):
    print(f'  msg[{i}] role={msg["role"]}, content length={len(msg["content"])}')
print(f'rubrics type: {type(sample["rubrics"])}')
print(f'rubrics count: {len(sample["rubrics"])}')
if sample["rubrics"]:
    print(f'rubrics[0]: {sample["rubrics"][0][:200]}')
print(f'metadata: {sample["metadata"]}')

# Count all categories and sub_categories
categories = Counter()
sub_categories = Counter()
cat_to_subs = {}

for item in ds:
    meta = item['metadata']
    if isinstance(meta, str):
        meta = json.loads(meta)
    cat = meta.get('context_category', 'unknown')
    sub = meta.get('sub_category', 'unknown')
    categories[cat] += 1
    sub_categories[sub] += 1
    if cat not in cat_to_subs:
        cat_to_subs[cat] = Counter()
    cat_to_subs[cat][sub] += 1

print(f'\n{"="*60}')
print(f'Context Categories ({len(categories)})')
print(f'{"="*60}')
for cat, count in categories.most_common():
    print(f'\n  {cat}: {count} samples')
    for sub, sub_count in cat_to_subs[cat].most_common():
        print(f'    - {sub}: {sub_count}')

# Find "Procedural" related samples
print(f'\n{"="*60}')
print(f'Looking for Procedural Task Execution samples...')
print(f'{"="*60}')

procedural_samples = []
for i, item in enumerate(ds):
    meta = item['metadata']
    if isinstance(meta, str):
        meta = json.loads(meta)
    cat = meta.get('context_category', '')
    sub = meta.get('sub_category', '')
    if 'procedur' in cat.lower() or 'procedur' in sub.lower():
        procedural_samples.append((i, item, meta))

print(f'Found {len(procedural_samples)} procedural samples')

if procedural_samples:
    # Show first 3 procedural samples in detail
    for idx, (i, item, meta) in enumerate(procedural_samples[:3]):
        print(f'\n--- Procedural Sample {idx} (index {i}) ---')
        print(f'metadata: {meta}')
        print(f'messages count: {len(item["messages"])}')
        print(f'rubrics count: {len(item["rubrics"])}')
        
        # Show messages
        for j, msg in enumerate(item['messages']):
            content_preview = msg['content'][:500].replace('\n', '\\n')
            print(f'  msg[{j}] role={msg["role"]}:')
            print(f'    {content_preview}...')
        
        # Show first 5 rubrics
        print(f'  rubrics (first 5):')
        for r in item['rubrics'][:5]:
            print(f'    - {r[:200]}')
else:
    # If no "procedural" keyword, show all unique categories
    print("\nNo samples with 'procedural' in category name.")
    print("Let me show a sample from each category:")
    seen_cats = set()
    for i, item in enumerate(ds):
        meta = item['metadata']
        if isinstance(meta, str):
            meta = json.loads(meta)
        cat = meta.get('context_category', '')
        if cat not in seen_cats:
            seen_cats.add(cat)
            print(f'\n--- Category: {cat} (sample index {i}) ---')
            print(f'  sub_category: {meta.get("sub_category", "")}')
            print(f'  messages count: {len(item["messages"])}')
            for j, msg in enumerate(item['messages'][:2]):
                content_preview = msg['content'][:300].replace('\n', '\\n')
                print(f'  msg[{j}] role={msg["role"]}:')
                print(f'    {content_preview}')
            print(f'  rubrics (first 3):')
            for r in item['rubrics'][:3]:
                print(f'    - {r[:200]}')

# Also save all data locally for further processing
print(f'\n{"="*60}')
print(f'Saving dataset locally...')
output_path = '/Users/leo/Desktop/ace/eval/CLbench/data/clbench_full.jsonl'
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    for item in ds:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f'Saved {len(ds)} samples to {output_path}')
print('Done!')
