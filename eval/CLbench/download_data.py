#!/usr/bin/env python3
"""Download CL-bench dataset and extract Rule System Application samples."""
import json
import os

print("Downloading CL-bench dataset from HuggingFace...")
from datasets import load_dataset

ds = load_dataset('tencent/CL-bench', split='train')
print(f'Total samples: {len(ds)}')
print(f'Columns: {ds.column_names}')

# Explore structure
from collections import Counter
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
print(f'All Categories and Sub-Categories')
print(f'{"="*60}')
for cat, count in categories.most_common():
    print(f'\n  {cat}: {count} samples')
    for sub, sub_count in cat_to_subs[cat].most_common():
        print(f'    - {sub}: {sub_count}')

# Extract Rule System Application samples
print(f'\n{"="*60}')
print(f'Extracting Rule System Application samples...')
print(f'{"="*60}')

rule_system_samples = []
for i, item in enumerate(ds):
    meta = item['metadata']
    if isinstance(meta, str):
        meta = json.loads(meta)
    cat = meta.get('context_category', '')
    if 'rule' in cat.lower() and 'system' in cat.lower():
        rule_system_samples.append({
            'messages': item['messages'],
            'rubrics': item['rubrics'],
            'metadata': meta if isinstance(meta, dict) else json.loads(meta)
        })

print(f'Found {len(rule_system_samples)} Rule System Application samples')

# Show sub-categories within Rule System Application
rule_subs = Counter()
for s in rule_system_samples:
    rule_subs[s['metadata']['sub_category']] += 1
print(f'Sub-categories:')
for sub, count in rule_subs.most_common():
    print(f'  - {sub}: {count}')

# Show a few sample structures
for idx in range(min(3, len(rule_system_samples))):
    sample = rule_system_samples[idx]
    print(f'\n--- Sample {idx} ---')
    print(f'  metadata: {sample["metadata"]}')
    print(f'  messages count: {len(sample["messages"])}')
    print(f'  rubrics count: {len(sample["rubrics"])}')
    for j, msg in enumerate(sample['messages']):
        content = msg['content']
        print(f'  msg[{j}] role={msg["role"]}, length={len(content)}')
        # Show first 500 chars
        preview = content[:500].replace('\n', '\\n')
        print(f'    preview: {preview}')
    print(f'  rubrics (first 5):')
    for r in sample['rubrics'][:5]:
        print(f'    - {r[:200]}')

# Save Rule System Application samples
output_dir = '/Users/leo/Desktop/ace/eval/CLbench/data'
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'rule_system_all.jsonl')
with open(output_path, 'w', encoding='utf-8') as f:
    for sample in rule_system_samples:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
print(f'\nSaved {len(rule_system_samples)} samples to {output_path}')

# Also save per sub-category
for sub in rule_subs:
    sub_samples = [s for s in rule_system_samples if s['metadata']['sub_category'] == sub]
    safe_name = sub.lower().replace(' ', '_').replace('/', '_')
    sub_path = os.path.join(output_dir, f'rule_system_{safe_name}.jsonl')
    with open(sub_path, 'w', encoding='utf-8') as f:
        for s in sub_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f'  Saved {len(sub_samples)} {sub} samples to {sub_path}')

print('\nDone!')
