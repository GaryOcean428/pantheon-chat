"""
Seed Patterns for Tool Factory Bootstrap

These patterns provide initial knowledge for the Tool Factory to generate tools.
Without seed patterns, the factory has no foundation to build upon.

QIG Philosophy: These are minimal seeds - the system should learn more patterns
through research, user templates, and observation. Seeds just bootstrap the process.
"""

from typing import List, Dict

# Seed patterns organized by category
SEED_PATTERNS: List[Dict] = [
    # =========================================================================
    # TEXT PROCESSING PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_text_transform",
        "source_type": "user_provided",
        "description": "Transform text to uppercase, lowercase, or title case",
        "code_snippet": '''def transform_text(text: str, mode: str = "upper") -> str:
    """Transform text case based on mode."""
    if mode == "upper":
        return text.upper()
    elif mode == "lower":
        return text.lower()
    elif mode == "title":
        return text.title()
    return text
''',
        "input_signature": {"text": "str", "mode": "str"},
        "output_type": "str"
    },
    {
        "pattern_id": "seed_text_extract",
        "source_type": "user_provided",
        "description": "Extract words, sentences, or patterns from text",
        "code_snippet": '''def extract_from_text(text: str, pattern: str = "words") -> list:
    """Extract elements from text based on pattern type."""
    import re
    if pattern == "words":
        return text.split()
    elif pattern == "sentences":
        return re.split(r'[.!?]+', text)
    elif pattern == "numbers":
        return re.findall(r'\\d+', text)
    elif pattern == "emails":
        return re.findall(r'[\\w.+-]+@[\\w-]+\\.[\\w.-]+', text)
    return [text]
''',
        "input_signature": {"text": "str", "pattern": "str"},
        "output_type": "list"
    },
    {
        "pattern_id": "seed_text_count",
        "source_type": "user_provided",
        "description": "Count words, characters, or lines in text",
        "code_snippet": '''def count_text(text: str, unit: str = "words") -> int:
    """Count elements in text."""
    if unit == "words":
        return len(text.split())
    elif unit == "chars":
        return len(text)
    elif unit == "lines":
        return len(text.splitlines())
    return 0
''',
        "input_signature": {"text": "str", "unit": "str"},
        "output_type": "int"
    },
    
    # =========================================================================
    # LIST/COLLECTION PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_list_filter",
        "source_type": "user_provided",
        "description": "Filter items in a list based on condition",
        "code_snippet": '''def filter_items(items: list, condition: str = "truthy") -> list:
    """Filter list items based on condition."""
    if condition == "truthy":
        return [x for x in items if x]
    elif condition == "numeric":
        return [x for x in items if isinstance(x, (int, float))]
    elif condition == "strings":
        return [x for x in items if isinstance(x, str)]
    elif condition == "unique":
        seen = set()
        result = []
        for x in items:
            key = str(x)
            if key not in seen:
                seen.add(key)
                result.append(x)
        return result
    return items
''',
        "input_signature": {"items": "list", "condition": "str"},
        "output_type": "list"
    },
    {
        "pattern_id": "seed_list_transform",
        "source_type": "user_provided",
        "description": "Transform list items - map, sort, reverse",
        "code_snippet": '''def transform_list(items: list, operation: str = "sort") -> list:
    """Transform list with operation."""
    if operation == "sort":
        return sorted(items, key=lambda x: str(x))
    elif operation == "reverse":
        return list(reversed(items))
    elif operation == "flatten":
        result = []
        for item in items:
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        return result
    elif operation == "uppercase":
        return [str(x).upper() for x in items]
    return items
''',
        "input_signature": {"items": "list", "operation": "str"},
        "output_type": "list"
    },
    {
        "pattern_id": "seed_list_aggregate",
        "source_type": "user_provided",
        "description": "Aggregate list values - sum, count, average",
        "code_snippet": '''def aggregate_list(items: list, operation: str = "sum") -> float:
    """Aggregate numeric list values."""
    numbers = [x for x in items if isinstance(x, (int, float))]
    if not numbers:
        return 0
    if operation == "sum":
        return sum(numbers)
    elif operation == "count":
        return len(numbers)
    elif operation == "avg":
        return sum(numbers) / len(numbers)
    elif operation == "max":
        return max(numbers)
    elif operation == "min":
        return min(numbers)
    return 0
''',
        "input_signature": {"items": "list", "operation": "str"},
        "output_type": "float"
    },
    
    # =========================================================================
    # DICT/JSON PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_dict_extract",
        "source_type": "user_provided",
        "description": "Extract values from dictionary by key or path",
        "code_snippet": '''def extract_from_dict(data: dict, key: str) -> any:
    """Extract value from dict, supports nested paths with dots."""
    if not data:
        return None
    if '.' in key:
        parts = key.split('.')
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current
    return data.get(key)
''',
        "input_signature": {"data": "dict", "key": "str"},
        "output_type": "Any"
    },
    {
        "pattern_id": "seed_dict_transform",
        "source_type": "user_provided",
        "description": "Transform dictionary - filter keys, rename, merge",
        "code_snippet": '''def transform_dict(data: dict, operation: str = "keys") -> any:
    """Transform dictionary based on operation."""
    if operation == "keys":
        return list(data.keys())
    elif operation == "values":
        return list(data.values())
    elif operation == "items":
        return list(data.items())
    elif operation == "flatten":
        result = {}
        def flatten(d, prefix=''):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    flatten(v, key)
                else:
                    result[key] = v
        flatten(data)
        return result
    return data
''',
        "input_signature": {"data": "dict", "operation": "str"},
        "output_type": "Any"
    },
    
    # =========================================================================
    # STRING MANIPULATION PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_string_split",
        "source_type": "user_provided",
        "description": "Split string by delimiter, whitespace, or pattern",
        "code_snippet": '''def split_string(text: str, delimiter: str = None) -> list:
    """Split string by delimiter."""
    if delimiter is None:
        return text.split()
    return text.split(delimiter)
''',
        "input_signature": {"text": "str", "delimiter": "str"},
        "output_type": "list"
    },
    {
        "pattern_id": "seed_string_join",
        "source_type": "user_provided",
        "description": "Join list of strings with separator",
        "code_snippet": '''def join_strings(items: list, separator: str = " ") -> str:
    """Join list items with separator."""
    return separator.join(str(x) for x in items)
''',
        "input_signature": {"items": "list", "separator": "str"},
        "output_type": "str"
    },
    {
        "pattern_id": "seed_string_replace",
        "source_type": "user_provided",
        "description": "Replace substring or pattern in text",
        "code_snippet": '''def replace_in_string(text: str, old: str, new: str) -> str:
    """Replace occurrences of old with new in text."""
    return text.replace(old, new)
''',
        "input_signature": {"text": "str", "old": "str", "new": "str"},
        "output_type": "str"
    },
    
    # =========================================================================
    # NUMERIC/MATH PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_math_basic",
        "source_type": "user_provided",
        "description": "Basic math operations on numbers",
        "code_snippet": '''def math_operation(a: float, b: float, op: str = "add") -> float:
    """Perform basic math operation."""
    if op == "add":
        return a + b
    elif op == "sub":
        return a - b
    elif op == "mul":
        return a * b
    elif op == "div":
        return a / b if b != 0 else 0
    elif op == "mod":
        return a % b if b != 0 else 0
    elif op == "pow":
        return a ** b
    return 0
''',
        "input_signature": {"a": "float", "b": "float", "op": "str"},
        "output_type": "float"
    },
    {
        "pattern_id": "seed_math_stats",
        "source_type": "user_provided",
        "description": "Statistical calculations on list of numbers",
        "code_snippet": '''def calculate_stats(numbers: list) -> dict:
    """Calculate statistics for list of numbers."""
    if not numbers:
        return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}
    nums = [float(x) for x in numbers if isinstance(x, (int, float))]
    if not nums:
        return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}
    return {
        "count": len(nums),
        "sum": sum(nums),
        "avg": sum(nums) / len(nums),
        "min": min(nums),
        "max": max(nums)
    }
''',
        "input_signature": {"numbers": "list"},
        "output_type": "dict"
    },
    
    # =========================================================================
    # DATE/TIME PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_datetime_format",
        "source_type": "user_provided",
        "description": "Format datetime to string",
        "code_snippet": '''def format_datetime(dt_string: str, format: str = "%Y-%m-%d") -> str:
    """Format datetime string to specified format."""
    from datetime import datetime
    try:
        # Try common formats
        for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y", "%m/%d/%Y"]:
            try:
                dt = datetime.strptime(dt_string, fmt)
                return dt.strftime(format)
            except ValueError:
                continue
        return dt_string
    except Exception:
        return dt_string
''',
        "input_signature": {"dt_string": "str", "format": "str"},
        "output_type": "str"
    },
    
    # =========================================================================
    # VALIDATION PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_validate_email",
        "source_type": "user_provided",
        "description": "Validate email address format",
        "code_snippet": '''def validate_email(email: str) -> bool:
    """Check if string is valid email format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
''',
        "input_signature": {"email": "str"},
        "output_type": "bool"
    },
    {
        "pattern_id": "seed_validate_url",
        "source_type": "user_provided",
        "description": "Validate URL format",
        "code_snippet": '''def validate_url(url: str) -> bool:
    """Check if string is valid URL format."""
    import re
    pattern = r'^https?://[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}.*$'
    return bool(re.match(pattern, url))
''',
        "input_signature": {"url": "str"},
        "output_type": "bool"
    },
    
    # =========================================================================
    # JSON PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_json_parse",
        "source_type": "user_provided",
        "description": "Parse JSON string to dict/list",
        "code_snippet": '''def parse_json(json_string: str) -> any:
    """Parse JSON string to Python object."""
    import json
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return None
''',
        "input_signature": {"json_string": "str"},
        "output_type": "Any"
    },
    {
        "pattern_id": "seed_json_stringify",
        "source_type": "user_provided",
        "description": "Convert Python object to JSON string",
        "code_snippet": '''def to_json(data: any, pretty: bool = False) -> str:
    """Convert Python object to JSON string."""
    import json
    if pretty:
        return json.dumps(data, indent=2)
    return json.dumps(data)
''',
        "input_signature": {"data": "Any", "pretty": "bool"},
        "output_type": "str"
    },
    
    # =========================================================================
    # HASH/CRYPTO PATTERNS
    # =========================================================================
    {
        "pattern_id": "seed_hash_string",
        "source_type": "user_provided",
        "description": "Hash string with various algorithms",
        "code_snippet": '''def hash_string(text: str, algorithm: str = "sha256") -> str:
    """Hash string with specified algorithm."""
    import hashlib
    data = text.encode('utf-8')
    if algorithm == "md5":
        return hashlib.md5(data).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    elif algorithm == "sha512":
        return hashlib.sha512(data).hexdigest()
    return hashlib.sha256(data).hexdigest()
''',
        "input_signature": {"text": "str", "algorithm": "str"},
        "output_type": "str"
    },
]


def get_seed_patterns() -> List[Dict]:
    """Return all seed patterns."""
    return SEED_PATTERNS


def get_seed_pattern_count() -> int:
    """Return count of seed patterns."""
    return len(SEED_PATTERNS)
