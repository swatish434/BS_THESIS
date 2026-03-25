import sys
# Add exp 14 as starting point
sys.argv = ['run_matrix_patches.py']

# Import everything from run_matrix_patches
exec(open('run_matrix_patches.py').read().replace(
    'for exp in matrix:',
    'matrix = [e for e in matrix if e["id"] >= 14]\nfor exp in matrix:'
))
