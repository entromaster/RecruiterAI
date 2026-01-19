# import re
# import json

# def preprocess_text(text):
#     # Preprocess the text by inserting spaces around "and" if it concatenates form numbers
#     text = re.sub(r'(\(\d{2}/\d{2}\))and(\w)', r'\1 and \2', text)
#     return text

# def extract_text_from_adobe_json(data):
#     text_list = []

#     def extract_text(item):
#         # If 'Text' exists in the current level, add it to the list
#         if 'Text' in item:
#             text_list.append(item['Text'])
#         # If 'Kids' exists, recursively call extract_text on each child
#         if 'Kids' in item:
#             for kid in item['Kids']:
#                 extract_text(kid)

#     # Process each top-level item in the data list
#     for entry in data:
#         extract_text(entry)

#     # Join all extracted text with newline characters and return
#     # return "\n".join(text_list)
#     return text_list

# def filter_adobe_json_by_object_id(adobe_json, start_index=None, end_index=None):
#     # Extract elements from adobe_json

#     # Set start_index to minimum ObjectID if None
#     if start_index is None:
#         start_index = min(item.get("ObjectID", float('inf')) for item in adobe_json)

#     # Set end_index to maximum ObjectID if None
#     if end_index is None:
#         end_index = max(item.get("ObjectID", float('-inf')) for item in adobe_json)

#     # Filter adobe_json list based on the adjusted ObjectID range
#     filtered_list = [
#         item for item in adobe_json
#         if start_index <= item.get("ObjectID", float('inf')) <= end_index
#     ]

#     return filtered_list

# # TODO: will take json not filepath and take index in function.
# def extract_form_numbers_from_json(adobe_json: dict, index: tuple):
#     # Open the PDF file
#     # with open(json_file_path, 'rb') as file:
#     # adobe_json = json.load(file)
#     adobe_json = adobe_json["elements"]
#     text_list = extract_text_from_adobe_json(filter_adobe_json_by_object_id(adobe_json, index[0], index[1]))

#     # Define the regex pattern to extract form numbers
#     # Handles patterns like JU1000 (05-17), LSW1135B (06/03), EZ-1(05/15)
#     # print(text)
#     form_number_pattern = r'[A-Z]+-\d{1,4}\(\d{2}/\d{2}\)|[A-Z]+\d{3,4}[A-Z]*\s\(\d{2}/\d{2}\)|[A-Z]+\d{3,4}[A-Z]*\s\(\d{2}-\d{2}\)|[A-Z]+\d{3,4}[A-Z]*\(\d{2}/\d{2}\)'
#     # form_number_pattern = r'[A-Z]{3}\s?\d{4,}\S*'

#     # Preprocess the text to handle concatenated form numbers
#     form_numbers = []
#     for text in text_list:
#         text = preprocess_text(text)
#         # Find all matches in the preprocessed text
#         matched_form_numbers = re.findall(form_number_pattern, text)
#         # print(form_numbers)
#         # Remove duplicates and sort to maintain order
#         matched_form_numbers = sorted(set(matched_form_numbers), key=matched_form_numbers.index)
#         form_numbers.extend(matched_form_numbers)


#         # Return the unique list of form numbers
#         return form_numbers

# def normalize_pattern(s):
#     # Use regular expressions to remove spaces and parentheses
#     return re.sub(r'[ ()-]', '', s)

# def extractFormNumber(adobe_json, index, client, AZURE_OPENAI_CHATGPT_DEPLOYMENT):
#     # Extract form numbers from each PDF
#     form_numbers = extract_form_numbers_from_json(adobe_json, index)
#     normalizeFormNumber = [normalize_pattern(i) for i in form_numbers]
#     return form_numbers, normalizeFormNumber


# import re
# import PyPDF2

# def preprocess_text(text):
#     # Preprocess the text by inserting spaces around "and" if it concatenates form numbers
#     text = re.sub(r'(\(\d{2}/\d{2}\))and(\w)', r'\1 and \2', text)
#     return text

# def extract_form_numbers_from_pdf(pdf_file_path):
#     # Open the PDF file
#     with open(pdf_file_path, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         num_pages = len(reader.pages)
#         text = ''

#         # Extract text from all the pages
#         for page_num in range(num_pages):
#             page = reader.pages[page_num]
#             text += page.extract_text()

#         # Preprocess the text to handle concatenated form numbers
#         text = preprocess_text(text)

#         # Define the regex pattern to extract form numbers
#         # Handles patterns like JU1000 (05-17), LSW1135B (06/03), EZ-1(05/15)
#         form_number_pattern = r'[A-Z]+-\d{1,4}\(\d{2}/\d{2}\)|[A-Z]+\d{3,4}[A-Z]*\s\(\d{2}/\d{2}\)|[A-Z]+\d{3,4}[A-Z]*\s\(\d{2}-\d{2}\)|[A-Z]+\d{3,4}[A-Z]*\(\d{2}/\d{2}\)'

#         # Find all matches in the preprocessed text
#         form_numbers = re.findall(form_number_pattern, text)

#         # Remove duplicates and sort to maintain order
#         form_numbers = sorted(set(form_numbers), key=form_numbers.index)

#         # Return the unique list of form numbers
#         return form_numbers

# def normalize_pattern(s):
#     # Use regular expressions to remove spaces and parentheses
#     return re.sub(r'[ ()-]', '', s)

# def extractFormNumber(file_path, client, AZURE_OPENAI_CHATGPT_DEPLOYMENT):
#     # Extract form numbers from each PDF
#     form_numbers = extract_form_numbers_from_pdf(file_path)
#     normalizeFormNumber = [normalize_pattern(i) for i in form_numbers]
#     print(f"Form numbers extracted from {file_path}:")
#     return form_numbers, normalizeFormNumber


import re
from monads.monad_class import monad_wrapper


@monad_wrapper
def preprocess_text(text):
    # Preprocess the text by inserting spaces around "and" if it concatenates form numbers
    text = re.sub(r"(\(\d{2}/\d{2}\))and(\w)", r"\1 and \2", text)
    return text


@monad_wrapper
def extract_text_from_adobe_json(data):
    text_list = []

    def extract_text(item):
        # If 'Text' exists in the current level, add it to the list
        if "Text" in item:
            text_list.append(item["Text"])
        # If 'Kids' exists, recursively call extract_text on each child
        if "Kids" in item:
            for kid in item["Kids"]:
                extract_text(kid)

    # Process each top-level item in the data list
    for entry in data:
        extract_text(entry)

    # Join all extracted text with newline characters and return
    # return "\n".join(text_list)
    # print(text_list)
    return text_list


@monad_wrapper
def filter_adobe_json_by_object_id(adobe_json, start_index=None, end_index=None):
    # Extract elements from adobe_json
    # adobe_json = adobe_json.get("elements", [])

    # Set start_index to minimum ObjectID if None
    if start_index is None:
        start_index = 0

    # Set end_index to maximum ObjectID if None
    if end_index is None:
        end_index = -1

    # Filter adobe_json list based on the adjusted ObjectID range
    filtered_list = adobe_json[start_index:end_index]

    # print(filtered_list)

    return filtered_list


@monad_wrapper
def extract_form_numbers_from_json(adobe_json: dict, index: tuple, form_number_pattern):
    # Open the PDF file
    # with open(json_file_path, 'rb') as file:
    # adobe_json = json.load(file)
    adobe_json = adobe_json["elements"]
    #print(adobe_json)
    text_list = extract_text_from_adobe_json(filter_adobe_json_by_object_id(adobe_json, index[0], index[1]).unwrap()).unwrap()

    # Define the regex pattern to extract form numbers
    # Handles patterns like JU1000 (05-17), LSW1135B (06/03), EZ-1(05/15)
    # print(text)
    # form_number_pattern = r'[A-Z]+-\d{1,4}\(\d{2}/\d{2}\)|[A-Z]+\d{3,4}[A-Z]*\s\(\d{2}/\d{2}\)|[A-Z]+\d{3,4}[A-Z]*\s\(\d{2}-\d{2}\)|[A-Z]+\d{3,4}[A-Z]*\(\d{2}/\d{2}\)'
    #form_number_pattern = r"^[A-Z]+-(?:[A-Z]{2,5}|[0-9]{2,5})-[A-Z]+(?:-[0-9]{3})?$"

    compiledregex = re.compile(form_number_pattern)
    #print("Compiled Regex: ", compiledregex)

    # Preprocess the text to handle concatenated form numbers
    form_numbers = []
    for text in text_list:
        text = preprocess_text(text).unwrap()
        # Find all matches in the preprocessed text
        matchedlist = compiledregex.findall(text.strip())
        # print(form_numbers)
        # Remove duplicates and sort to maintain order
        if matchedlist:
            form_numbers.append(text.strip())

    # Return the unique list of form numbers
    return form_numbers


@monad_wrapper
def normalize_pattern(s):
    # Use regular expressions to remove spaces and parentheses
    return re.sub(r"[ ()-]", "", s)


@monad_wrapper
def extractFormNumber(adobe_json, index, client, AZURE_OPENAI_CHATGPT_DEPLOYMENT, regex_pattern):
    # Extract form numbers from each PDF
    form_numbers = extract_form_numbers_from_json(adobe_json, index, regex_pattern).unwrap()
    print("Form Numbers: ",form_numbers )
    normalizeFormNumber = [normalize_pattern(i).unwrap() for i in form_numbers]
    return form_numbers, normalizeFormNumber
