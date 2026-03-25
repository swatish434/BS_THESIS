import json
import sys

def extract_code(nb_path, py_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(py_path, 'w', encoding='utf-8') as f:
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                for line in source:
                    f.write(line)
                f.write('\n\n')

if __name__ == '__main__':
    extract_code(sys.argv[1], sys.argv[2])
