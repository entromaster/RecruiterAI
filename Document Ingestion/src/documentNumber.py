import re
from datetime import datetime
from typing import Union, Dict, List, Tuple, Optional
from monads.monad_class import monad_wrapper
import fitz
import base64
from PIL import Image
from io import BytesIO
import json
import os

def extract_pages_as_base64_images(pdf_file_path: str, max_pages: int = 2) -> List[str]:
    """Extract initial pages from PDF as base64 encoded images."""
    pdf_document = fitz.open(pdf_file_path)
    num_pages = min(len(pdf_document), max_pages)
    base64_images = []
    
    for page_num in range(num_pages):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        img_byte_array = BytesIO()
        img.save(img_byte_array, format='JPEG')
        img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('ascii')
        base64_images.append(f"data:image/jpeg;base64,{img_base64}")
    
    pdf_document.close()
    return base64_images

def vision_response_generator(client, deployment, base64_images: List[str], regex_patterns: Union[re.Pattern, List[re.Pattern]]) -> Optional[Dict]:
    """Generate vision API response for document images."""
    # Convert patterns to string format for prompt
    patterns = regex_patterns if isinstance(regex_patterns, list) else [regex_patterns]
    pattern_strings = [p.pattern for p in patterns]
    patterns_text = "\n".join(f"Pattern {i+1}: {p}" for i, p in enumerate(pattern_strings))
    
    system_message = {
        "role": "system",
        "content": [{
            "type": "text",
            "text": f"""
            Analyze the document image and extract the document/form number that matches the following patterns:
            {patterns}
            
            Key instructions:
            1. Look for numbers/identifiers that match these specific patterns
            2. Focus on the main content area, not footer numbers
            3. Return only the document number without additional text
            4. If multiple matching patterns are found, return all potential matches
            5. Format: List each match on a new line, prefixed with "MATCH: " 
            
            Example output:
            MATCH: F123-45(01/2024)
            MATCH: Form-ABC-99-2024
            """
        }]
    }
    
    user_content = [{"type": "image_url", "image_url": {"url": img}} for img in base64_images]
    
    messages = [
        system_message,
        {"role": "user", "content": user_content}
    ]
    
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=1000,
            temperature=0.3,
            top_p=0.95,
        )
        return response
    except Exception as e:
        print(f"Vision API error: {str(e)}")
        return None

@monad_wrapper
def process_vision_response(response: Dict, regex_patterns: Union[re.Pattern, List[re.Pattern]]) -> str:
    print(response)
    """
    Extract document number from vision API response, validating against regex patterns.
    """
    print("DEBUG process_vision_response: Starting")
    
    # Split the content into lines and parse each line
    content = response.choices[0].message.content.strip()
    
    print(f"DEBUG process_vision_response: Raw content: {content}")
    matches = []
    
    # Extract all matches from the response
    for line in content.split('\n'):
        if line.startswith('MATCH:'):
            potential_match = line.replace('MATCH:', '').strip()
            matches.append(potential_match)
            
    print(f"DEBUG process_vision_response: Found matches: {matches}")
    
    if not matches:
        raise ValueError("No matches found in vision response")
    
    # Validate matches against regex patterns
    patterns = regex_patterns if isinstance(regex_patterns, list) else [regex_patterns]
    validated_matches = []
    
    for match in matches:
        for pattern in patterns:
            if pattern.match(match):
                validated_matches.append(match)
                break
    
    print(f"DEBUG process_vision_response: Validated matches: {validated_matches}")
    
    if not validated_matches:
        raise ValueError("No valid matches found after pattern validation")
    
    result = validated_matches[0]
    print(f"DEBUG process_vision_response: Selected match: {result}")
    return result  # This will be wrapped in a Monad by the decorator




@monad_wrapper
def preprocess_text(text: str) -> str:
    """Preprocess the input text."""
    return text.strip()

@monad_wrapper
def convert_format_to_strptime(fmt: str) -> str:
    """Convert custom date format to strptime format"""
    replacements = {
        'MM': '%m',
        'DD': '%d',
        'YYYY': '%Y',
        'YY': '%y'
    }
    print(f"DEBUG convert_format: Input format: {fmt}")
    for custom, strp in replacements.items():
        fmt = fmt.replace(custom, strp)
    print(f"DEBUG convert_format: Output format: {fmt}")
    return fmt

@monad_wrapper
def get_format_granularity(format_str: str) -> int:
    """Calculate the granularity of a date format string."""
    DATE_COMPONENTS = {
        'MM': 2,
        'DD': 2,
        'YYYY': 4,
        'YY': 2
    }
    total = sum(DATE_COMPONENTS[comp] for comp in DATE_COMPONENTS if comp in format_str)
    print(f"DEBUG granularity: Format {format_str} has granularity {total}")
    return total

@monad_wrapper
def standardize_document_date(doc_number: str, date_config: dict) -> str:
    """
    Standardize the date in a document number if needed based on granularity comparison.
    """
    try:
        if not date_config:
            return doc_number

        footer_config = date_config.get('footer', {})
        table_config = date_config.get('table', {})

        footer_pattern = footer_config.get('pattern')
        footer_format = footer_config.get('format')
        table_format = table_config.get('format')

        if not (footer_pattern and footer_format and table_format):
            return doc_number

        print(f"DEBUG: Looking for pattern {footer_pattern} in {doc_number}")
        match = re.search(footer_pattern, doc_number)
        if not match:
            return doc_number

        full_match = match.group(0)
        date_str = match.group(1)
        start, end = match.span(0)

        form_number_part = doc_number[:start].rstrip()

        footer_granularity = get_format_granularity(footer_format).unwrap()
        table_granularity = get_format_granularity(table_format).unwrap()

        if footer_granularity > table_granularity:
            try:
                footer_strptime = convert_format_to_strptime(footer_format).unwrap()
                table_strptime = convert_format_to_strptime(table_format).unwrap()

                date_obj = datetime.strptime(date_str, footer_strptime)
                standardized_date = date_obj.strftime(table_strptime)

                if full_match.startswith('(') and full_match.endswith(')'):
                    standardized_date = f"({standardized_date})"

                return f"{form_number_part} {standardized_date}"

            except (ValueError, TypeError) as e:
                print(f"Error standardizing date: {str(e)}")
                return doc_number

        date_part = doc_number[start:].lstrip()
        return f"{form_number_part} {date_part}"

    except Exception as e:
        print(f"Error in date standardization: {str(e)}")
        return doc_number

@monad_wrapper
def adjust_regex_anchors_and_find_first_largest_match(text: str, regex: Union[str, re.Pattern]) -> Optional[str]:
    """
    Adjusts regex by removing start/end anchors and finds the first largest matching group.
    """
    print(f"\nDEBUG adjust_regex:")
    print(f"Input text: {text}")
    print(f"Input regex: {regex}")

    if isinstance(regex, re.Pattern):
        regex = regex.pattern

    if regex.startswith('^'):
        regex = regex[1:]
    if regex.endswith('$'):
        regex = regex[:-1]

    print(f"DEBUG: Adjusted regex: {regex}")

    try:
        matches = re.findall(regex, text)
        print(f"DEBUG: Found matches: {matches}")

        if not matches:
            return None

        max_length = max(len(match) for match in matches)
        for match in matches:
            if len(match) == max_length:
                print(f"DEBUG: Selected longest match: {match}")
                return match

    except Exception as e:
        print(f"DEBUG: Error in regex adjustment: {str(e)}")
        return None

    return None
    

@monad_wrapper
def extract_document_number(text: str, regex_patterns: Union[re.Pattern, List[re.Pattern]]) -> str:
    """Extract document number using regex patterns."""
    print(f"\nDEBUG extract_document_number:")
    print(f"Input text: {text}")

    text = preprocess_text(text).unwrap()
    patterns = [regex_patterns] if isinstance(regex_patterns, re.Pattern) else regex_patterns

    for pattern in patterns:
        print(f"DEBUG: Trying pattern: {pattern.pattern}")
        matched_list = pattern.findall(text)
        if matched_list:
            print(f"DEBUG: Found matches: {matched_list}")
            return matched_list[0]

    for pattern in patterns:
        print(f"DEBUG: Trying pattern with anchor adjustment: {pattern.pattern}")
        fallback_match = adjust_regex_anchors_and_find_first_largest_match(text, pattern).unwrap()
        if fallback_match:
            print(f"DEBUG: Found fallback match: {fallback_match}")
            return fallback_match

    raise ValueError("Document number not found")

@monad_wrapper
def normalize_pattern(s: str) -> str:
    """Remove spaces, parentheses, and other special characters."""
    original = s
    normalized = re.sub(r"[ ()-/]", "", s)
    print(f"DEBUG normalize_pattern: {original} -> {normalized}")
    return normalized

@monad_wrapper
def extractDocumentNumber(
    split_input: Union[str, Dict],
    regex_patterns: Union[re.Pattern, List[re.Pattern]],
    config: Optional[dict] = None,
    pdf_path: Optional[str] = None,
    client: Optional[object] = None,
    deployment: Optional[str] = None
) -> Tuple[str, str]:
    """
    Enhanced document number extraction with PDF fallback support.
    
    Args:
        split_input: Text input as string or dict
        regex_patterns: Regex patterns for extraction
        config: Configuration for date standardization
        pdf_path: Optional path to PDF file
        client: Optional API client for vision service
        deployment: Optional deployment name for vision service
        
    Returns:
        Tuple of (original_number, normalized_number)
    """
    try:
        # Attempt conventional extraction first
        if isinstance(split_input, str) and split_input:
            document_number = split_input
        elif isinstance(split_input, dict) and any(split_input.values()):
            text = "\n".join(split_input.values())
            document_number = extract_document_number(text, regex_patterns).unwrap()
        elif pdf_path and client and deployment:
            # Fallback to PDF processing
            print("DEBUG: Falling back to PDF processing")
            base64_images = extract_pages_as_base64_images(pdf_path)
            vision_response = vision_response_generator(client, deployment, base64_images, regex_patterns)
            document_number = process_vision_response(vision_response, regex_patterns).unwrap()
        else:
            raise ValueError("No valid input provided")
        
        # Apply date standardization if configured
        if config and 'DateSubConfig' in config:
            document_number = standardize_document_date(document_number, config['DateSubConfig']).unwrap()
        
        # If document_number is a Monad, unwrap it
        if hasattr(document_number, 'unwrap'):
            document_number = document_number.unwrap()
            
        normalized_document_number = normalize_pattern(document_number).unwrap()
        
        # Return the unwrapped values as a tuple
        return (document_number, normalized_document_number)
        
    except Exception as e:
        print(f"Error in document number extraction: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Document number type: {type(document_number) if 'document_number' in locals() else 'Not created'}")
        raise ValueError(f"Failed to extract document number: {str(e)}")