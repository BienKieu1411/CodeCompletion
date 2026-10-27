
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tree-sitter-python', 'tree-sitter-java'])
print("Tree-sitter python and java installed successfully!")
