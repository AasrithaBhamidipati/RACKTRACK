import json, sys
fp = sys.argv[1]
search = sys.argv[2]
with open(fp, 'r', encoding='utf-8') as f:
    data = json.load(f)
found = []

def walk(node):
    if isinstance(node, dict):
        if 'segmented_filename' in node and node['segmented_filename'].endswith(search):
            found.append(node)
        for v in node.values():
            walk(v)
    elif isinstance(node, list):
        for i in node:
            walk(i)

walk(data)
print('Matches:', len(found))
for n in found:
    print('---')
    print('segmented_filename=', n.get('segmented_filename'))
    print('parent_type=', n.get('parent_type'))
    print('parent_id=', n.get('parent_id'))
    print('parent_image=', n.get('parent_image'))
