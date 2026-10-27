import re
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GET_API():
    def __init__(self, examples, language):
        self.examples = examples
        self.language = language

    def get(self):
        examples_api = []
        for example in self.examples:
            if self.language == 'python':
                api_info = self._process_python_example(example)
            elif self.language == 'java':
                api_info = self._process_java_example(example)
            else:
                api_info = ""
            examples_api.append(api_info)
        return examples_api
    
    def _process_python_example(self, example):
        
        imported_entities = self._extract_python_imports(example.left_context)
        api_descriptions = []
        for entity_info in imported_entities:
            entity_name = entity_info['name']
            definition_info = self._find_python_definition(entity_name, example.related_files)
            if definition_info:
                if definition_info['type'] == 'class':
                    api_descriptions.append(definition_info['content'])
                elif definition_info['type'] == 'function':
                    api_descriptions.append(definition_info['signature'])
    
        return "\n\n".join(api_descriptions)
    
    def _extract_python_imports(self, content):
        
        imported_entities = []
        std_libs = {
            'abc', 'argparse', 'asyncio', 'collections', 'concurrent', 'contextlib',
            'copy', 'csv', 'datetime', 'decimal', 'functools', 'glob', 'io', 'json',
            'logging', 'math', 'os', 'pathlib', 're', 'shutil', 'socket', 'subprocess',
            'sys', 'tempfile', 'threading', 'time', 'typing', 'unittest', 'uuid',
            'xml', 'zipfile'}
        
        third_party_libs = {
            'numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn', 'tensorflow', 
            'torch', 'requests', 'flask', 'django', 'bs4', 'selenium', 'pytest',
            'sqlalchemy', 'pydantic', 'fastapi', 'nltk', 'transformers', 'pillow',
            'openai', 'websockets', 'aiohttp', 'sentencepiece', 'tqdm', 'torchvision'}
        
        import_lines = re.findall(r'(?:^|\n)(?:import|from)\s+.*?(?:$|\n)', content, re.MULTILINE)
        for line in import_lines:
            line = line.strip()
            if line.startswith('from'):
                match = re.match(r'from\s+([a-zA-Z0-9_.]+)\s+import\s+(.*?)$', line)
                if match:
                    module = match.group(1)
                    base_module = module.split('.')[0]
                    if base_module in std_libs or base_module in third_party_libs:
                        continue
                    items = re.split(r',\s*', match.group(2))
                    for item in items:
                        item = item.strip()
                        if ' as ' in item:
                            original, alias = item.split(' as ', 1)
                            imported_entities.append({'name': original, 'module': module})
                        else:
                            imported_entities.append({'name': item, 'module': module})
            elif line.startswith('import'):
                items = re.split(r',\s*', line[6:].strip())
                for item in items:
                    item = item.strip()
                    base_module = item.split('.')[0].split(' as ')[0]
                    
                    if base_module in std_libs or base_module in third_party_libs:
                        continue
                    
                    if ' as ' in item:
                        original, alias = item.split(' as ', 1)
                        imported_entities.append({'name': original, 'module': original})
                    else:
                        parts = item.split('.')
                        imported_entities.append({'name': parts[0], 'module': item})
        
        return imported_entities
    

    def _find_python_definition(self, name, related_files):

        for file in related_files:
            class_pattern = rf'class\s+{re.escape(name)}\s*(?:\(.*?\))?\s*:'
            class_match = re.search(class_pattern, file.code_content, re.DOTALL)
            if class_match:
                class_content = self._parse_class_content(file.code_content, class_match)
                return {'type': 'class', 'content': class_content}
            
            func_pattern = rf'def\s+{re.escape(name)}\s*\(.*?\)\s*(?:->\s*.*?)?\s*:'
            func_match = re.search(func_pattern, file.code_content, re.MULTILINE)
            if func_match:
                func_sig = self._parse_function_signature(file.code_content, func_match)
                return {'type': 'function', 'signature': func_sig}
        
        return None
    
    def _parse_class_content(self, code_content, match):

        start_pos = match.start()
        header_line = code_content[start_pos: code_content.find('\n', start_pos)].strip()
        # logging.info(f'header line:\n{header_line}')
        lines = code_content[start_pos:].split('\n')
        if len(lines) < 2:
            return header_line

        body_lines = lines[1:]
        indent = None
        for line in body_lines:
            if line.strip():
                indent_match = re.match(r'^(\s+)', line)
                if indent_match:
                    indent = indent_match.group(1)
                    break
        
        if indent is None:
            return header_line
        
        indent_len = len(indent)
        method_signatures = []
        index = 0
        while index < len(body_lines):
            line = body_lines[index]
            index += 1

            if line.strip() == '':
                continue

            line_indent = re.match(r'^\s*', line).group(0)
            if len(line_indent) < indent_len:
                break

            if line.lstrip().startswith('class '):
                head_line = ' ' * 4 + line.strip()
                method_signatures_nested = []
                while index < len(body_lines):
                    line_temp = body_lines[index]
                    index += 1
                    if line_temp.strip() == '':
                        continue
                    line_temp_indent = re.match(r'^\s*', line_temp).group(0)
                    if line_temp_indent <= line_indent:
                        break
                    if line_temp.lstrip().startswith('def '):
                        sig_nested = line_temp.strip()
                        line_tmp = sig_nested
                        while index < len(body_lines) and (line_tmp.strip() == '' or line_tmp[-1] != ':'):
                            line_tmp = body_lines[index].strip()
                            sig_nested += line_tmp
                            index += 1
                        sig_nested = ' ' * 8 + sig_nested
                        method_signatures_nested.append(sig_nested)
                 
                nested_class_info = '\n'.join([head_line] + method_signatures_nested)
                method_signatures.append(nested_class_info)
                
            elif line.lstrip().startswith('def '):
                sig = line.strip()
                line_temp = sig
                while index < len(body_lines) and (line_temp.strip() == '' or line_temp[-1] != ':'):
                    line_temp = body_lines[index].strip()
                    sig += line_temp
                    index += 1
                sig = ' ' * 4 + sig
                method_signatures.append(sig)
        
        class_api_info = '\n'.join([header_line] + method_signatures)
        return class_api_info
    
    def _parse_function_signature(self, code_content, match):

        start = match.start()
        end = code_content.find('\n', start)
        end_1 = code_content.find(':', start)
        end = max(end, end_1)
        if end == -1:
            end = len(code_content)
        sig_line = code_content[start:end].strip()
        # logging.info(f'sig_line:\n{sig_line}')
        return sig_line
    
    def _parse_java_function_signature(self, code_content, match):
        start = match.start()
        end = code_content.find('{', start) + 1
        if end == 0:
            end = code_content.find(';', start) + 1
        if end == 0:
            end = len(code_content)
        signature = code_content[start:end].strip()
        if signature.endswith('{'):
            signature = signature.rstrip('{').strip()

        return signature

    def _extract_java_imports(self, content):
        imported_entities = []
        standard_packages = [
            'java.',
            'javax.',
            'javafx.',
            'com.sun.',
            'com.oracle.',
            'sun.',
            'jdk.',
            'oracle.'
        ]
        
        import_lines = re.findall(r'(?:^|\n)import\s+.*?;', content, re.MULTILINE)
        
        for line in import_lines:
            line = line.strip()
            if line.startswith('import'):
                package_path = line[6:].strip().rstrip(';').strip()
                
                is_standard_lib = False
                for std_pkg in standard_packages:
                    if package_path.startswith(std_pkg) or (
                        package_path.startswith('static ') and 
                        package_path[7:].startswith(std_pkg)
                    ):
                        is_standard_lib = True
                        break
                
                if is_standard_lib:
                    continue
                
                is_static = False
                if package_path.startswith('static '):
                    is_static = True
                    package_path = package_path[7:]
                
                if package_path.endswith('.*'):
                    package = package_path[:-2]
                    imported_entities.append({
                        'name': '*',
                        'original': package_path,
                        'package': package,
                        'is_static': is_static
                    })
                else:

                    parts = package_path.rsplit('.', 1)
                    if len(parts) > 1:
                        package, class_name = parts
                    else:
                        package, class_name = "", parts[0]
                        
                    imported_entities.append({
                        'name': class_name,
                        'original': package_path,
                        'package': package,
                        'is_static': is_static
                    })
        
        return imported_entities    

    def _process_java_example(self, example):

        processed_entities = set()
        imported_entities = self._extract_java_imports(example.left_context)
        api_descriptions = []
        
        for entity_info in imported_entities:
            entity_name = entity_info['name']
            
            if entity_name in processed_entities or entity_name == '*':
                continue
                
            processed_entities.add(entity_name)
            definition_info = self._find_java_definition(entity_name, example.related_files)
            
            if definition_info:
                if definition_info['type'] == 'class':
                    api_descriptions.append(definition_info['content'])
                elif definition_info['type'] == 'method':
                    method_sig = definition_info['signature']
                    api_descriptions.append(method_sig)
        
        for entity_info in imported_entities:
            if entity_info['name'] == '*':
                package_name = entity_info['package']
                package_classes = self._find_classes_in_package(package_name, example.related_files)
                
                for class_name in package_classes:
                    if class_name in processed_entities:
                        continue
                        
                    processed_entities.add(class_name)
                    definition_info = self._find_java_definition(class_name, example.related_files)
                    
                    if definition_info:
                        if definition_info['type'] == 'class':
                            api_descriptions.append(definition_info['content'])
                        elif definition_info['type'] == 'method':
                            method_sig = definition_info['signature']
                            api_descriptions.append(method_sig)
        
        return "\n\n".join(api_descriptions)

    def _find_classes_in_package(self, package_name, related_files):

        classes = []
        package_pattern = re.compile(rf'package\s+{re.escape(package_name)}\s*;', re.MULTILINE)
        class_pattern = re.compile(r'(?:public|private|protected|abstract|final)?\s+(?:class|interface|enum)\s+(\w+)', re.MULTILINE)
        
        for file in related_files:
            package_match = package_pattern.search(file.code_content)
            if package_match:
                class_matches = class_pattern.finditer(file.code_content)
                for match in class_matches:
                    class_name = match.group(1)
                    classes.append(class_name)
        
        return classes
    
    def _parse_java_class_content(self, code_content, match):

        start_pos = match.start()
        
        header_end = code_content.find('{', start_pos) + 1
        if header_end <= 0:
            return code_content[start_pos:].split('\n')[0].strip()
            
        header_line = code_content[start_pos:header_end].strip()
        
        end_pos = self._find_matching_brace_end(code_content, header_end - 1)
        if end_pos == -1:
            return header_line + " ... }"
        
        class_body = code_content[header_end:end_pos]
        
        method_pattern = r'\b(?:public|private|protected|static|final|abstract|synchronized)(?:\s+(?:public|private|protected|static|final|abstract|synchronized))*\s+(?!if|while|for|switch|catch)[\w<>\[\],\s.]+?\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w\s,\.]+)?\s*\{'
        method_matches = list(re.finditer(method_pattern, class_body))
        
        method_signatures = []
        for method_match in method_matches:
            method_start = method_match.start()

            brace_pos = class_body.find('{', method_start)
            if brace_pos == -1:
                continue
                
            method_sig = class_body[method_start:brace_pos].strip()
            method_signatures.append(' ' * 4 + method_sig)
        
        class_api_info = '\n'.join([header_line] + method_signatures) + '\n}'
        return class_api_info

    def _find_matching_brace_end(self, text, open_brace_pos):

        if open_brace_pos >= len(text) or text[open_brace_pos] != '{':
            return -1
            
        stack = ['{']
        pos = open_brace_pos + 1
        
        while pos < len(text) and stack:
            if text[pos] == '{':
                stack.append('{')
            elif text[pos] == '}':
                stack.pop()
                if not stack:
                    return pos
            pos += 1
        
        return -1

    def _find_java_definition(self, name, related_files):

        class_pattern = re.compile(
            rf'(?:public|private|protected|abstract|final)?\s+(?:class|interface|enum)\s+{re.escape(name)}\s*(?:extends\s+\w+(?:\.\w+)*)?(?:\s+implements\s+[\w\s,\.]+)?\s*\{{',
            re.DOTALL
        )
        
        method_pattern = re.compile(
            rf'(?:public|private|protected|static|final|abstract|synchronized)?\s+(?:[\w<>\[\],\s]+)\s+{re.escape(name)}\s*\([^)]*\)\s*(?:throws\s+[\w\s,\.]+)?\s*\{{',
            re.MULTILINE
        )
        
        for file in related_files:
            content = file.code_content
            
            class_match = class_pattern.search(content)
            if class_match:
                class_content = self._parse_java_class_content(content, class_match)
                return {'type': 'class', 'content': class_content}
            
            method_match = method_pattern.search(content)
            if method_match:
                method_sig = self._parse_java_function_signature(content, method_match)
                return {'type': 'method', 'signature': method_sig}
        
        return None