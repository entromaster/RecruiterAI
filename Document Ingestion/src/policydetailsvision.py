import fitz
import base64
from PIL import Image
from io import BytesIO
import json

def extract_pages_as_base64_images(pdf_file_path, max_pages=2):
    """
    Extract up to max_pages from the PDF and convert them to base64 encoded images.

    Args:
        pdf_file_path (str): Path to the PDF file
        max_pages (int): Maximum number of pages to extract (default: 10)

    Returns:
        list: List of base64 encoded images
    """
    # Open the PDF
    pdf_document = fitz.open(pdf_file_path)

    # Determine how many pages to process
    num_pages = min(len(pdf_document), max_pages)

    base64_images = []

    # Process each page
    for page_num in range(num_pages):
        # Extract the page
        page = pdf_document.load_page(page_num)

        # Convert the page to a pixmap (image object)
        pix = page.get_pixmap()

        # Convert the pixmap to an image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Save the image to a BytesIO stream
        img_byte_array = BytesIO()
        img.save(img_byte_array, format='JPEG')

        # Get the base64 encoded string
        img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('ascii')

        # Add the formatted base64 string to our list
        base64_images.append(f"data:image/jpeg;base64,{img_base64}")

    pdf_document.close()
    return base64_images

def response_generator(client, AZURE_OPENAI_CHATGPT_DEPLOYMENT, base64_images):
    """
    Generate responses for multiple base64 encoded images using GPT-4 Vision in a single request.

    Args:
        base64_images (list): List of base64 encoded images

    Returns:
        dict: Response from the API containing analysis for all images
    """

    # Create the system message
    system_message = {
        "role": "system",
        "content": [{
            "type": "text",
            "text": '''
            Please analyze the insurance policy document images and extract the following information in a structured format. Pay careful attention to these specific requirements:

1. Extract and format the following fields:
   - Policy Holder Name: [Extract the full name as shown]
   - Policy Start Date: [Format as MM/DD/YYYY]
   - Policy End Date: [Format as MM/DD/YYYY]
   - Declaration Number: [Look for this in the main body of the document, NOT in the footer]

2. Important Instructions:
   - All dates must be provided in MM/DD/YYYY format
   - The Declaration Number should be found in the main content area of the policy document
   - Do not confuse any footer numbers for the Declaration Number
   - If any field is not clearly visible or cannot be found, mark it as "Not Found"

3. Please provide the output in this exact format:
```
Policy Holder Name: [Name]
Policy Start Date: MM/DD/YYYY
Policy End Date: MM/DD/YYYY
Declaration Number: [Number]
```

4. Additional Notes:
   - If you find multiple potential values for any field, list all options and indicate which one you believe is correct and why
   - If dates are formatted differently in the document, please convert them to the required MM/DD/YYYY format
   - For the Policy Holder Name, include middle names/initials if present

Return only the requested information in the specified format without additional commentary.
            '''
        }]
    }

    # Create user message with all images
    user_content = []
    for base64_image in base64_images:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": base64_image
            }
        })

    messages = [
        system_message,
        {
            "role": "user",
            "content": user_content
        }
    ]

    # Payload for the request
    # messages = messages
    temperature = 0.7
    top_p = 0.95
    max_tokens = 4000

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response
    except Exception:
        try:
            # One retry attempt
            response = client.chat.completions.create(
                model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return response
        except Exception as retry_error:
            print(f"Failed to process request. Error: {retry_error}")
            return None

# Example usage:
def process_pdf_document(pdf_path):
    """
    Process a PDF document and extract document numbers from all pages.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        list: List of document numbers extracted from each page
    """
    try:
        # Extract base64 images from PDF
        base64_images = extract_pages_as_base64_images(pdf_path)

        # Get response for all images in a single request
        response = response_generator(base64_images)

        # Process response to extract document numbers
        document_numbers = []
        if response and 'choices' in response:
            # Split the content into lines and parse each line
            content = response['choices'][0]['message']['content'].strip()
            lines = content.split('\n')

            for line in lines:
                line = line.strip()
                if line:
                    # Extract page number and document number from the line
                    try:
                        page_str, doc_number = line.split(':', 1)
                        page_num = int(page_str.replace('Page', '').strip())
                        doc_number = doc_number.strip()

                        document_numbers.append({
                            'page': page_num,
                            'document_number': doc_number
                        })
                    except (ValueError, IndexError):
                        document_numbers.append({
                            'error': f'Failed to parse line: {line}'
                        })
        else:
            return [{'error': 'Failed to get response from API'}]

        return document_numbers

    except Exception as e:
        return [{'error': f'Failed to process PDF: {str(e)}'}]
    
# Define the function schema
FUNCTION_SCHEMA = {
    "name": "process_insurance_policy",
    "description": "Process and extract information from insurance policy text",
    "parameters": {
        "type": "object",
        "properties": {
            "policy_holder_name": {
                "type": "string",
                "description": "Full name of the policy holder"
            },
            "additional_insured": {
                "type": "string",
                "description": "Name of additional insured if present"
            },
            "policy_start_date": {
                "type": "string",
                "description": "Start date of the policy in MM/DD/YYYY"
            },
            "policy_end_date": {
                "type": "string",
                "description": "End date of the policy"
            },
            "declaration_number": {
                "type": "string",
                "description": "Policy declaration number in MM/DD/YYYY"
            }
        },
        "required": ["policy_holder_name", "policy_start_date", "policy_end_date", "declaration_number"]
    }
}

def parse_policy_information(policy_text, client, AZURE_OPENAI_CHATGPT_DEPLOYMENT):
    """
    Parse insurance policy information using OpenAI's function calling

    Args:
        policy_text (str): Raw policy text to be parsed

    Returns:
        Dict: Structured policy information
    """
    try:
        # Create the messages for the API call
        messages = [
            {
                "role": "system",
                "content": "You are a precise insurance policy information extractor. Extract the requested information from the policy text."
            },
            {
                "role": "user",
                "content": f"Extract the policy information from the following text:\n\n{policy_text}"
            }
        ]

        # Make the API call with function calling
        response = client.chat.completions.create(
            model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,  # Or your preferred model
            messages=messages,
            tools=[{"type": "function", "function": FUNCTION_SCHEMA}],
            tool_choice={"type": "function", "function": {"name": "process_insurance_policy"}}
        )

        # Extract the function call arguments
        if response.choices[0].message.tool_calls:
            function_args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return function_args
        else:
            raise ValueError("No function call in response")

    except Exception as e:
        print(f"Error parsing policy information: {str(e)}")
        return {}

def extract_and_select_page_images(pdf_file_path, page_num=None, max_items=2, max_images=10):
    """
    Extract up to max_images base64 images from the PDF and select page spans.

    Args:
        pdf_file_path (str): Path to the PDF file
        page_num (list or tuple, optional): Page number span(s). If None, take the first max_images pages.
        max_items (int): Maximum number of entries to select from each span
        max_images (int): Maximum total number of images to extract

    Returns:
        list: List of selected base64 encoded images
        list: Selected page spans (up to max_items and max_images)
    """
    pdf_document = fitz.open(pdf_file_path)

    # Default to the first max_images pages if page_num is not provided
    if page_num is None:
        page_num = [(i + 1, i + 1) for i in range(min(max_images, len(pdf_document)))]

    # Validate and select page spans
    selected_spans = []

    if isinstance(page_num, list):
        for span in page_num:
            if len(selected_spans) + 1 <= max_images:
                selected_spans.extend([span][:max_items])
            if len(selected_spans) >= max_images:
                break
    elif isinstance(page_num, tuple):
        selected_spans = [page_num]
    else:
        raise ValueError("Invalid page_num format. Must be list or tuple.")

    selected_spans = selected_spans[:max_images]

    base64_images = []

    for page_range in selected_spans:
        start_page, end_page = page_range
        for page_num in range(start_page - 1, end_page):  # fitz is 0-indexed
            if len(base64_images) >= max_images:
                break

            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            img_byte_array = BytesIO()
            img.save(img_byte_array, format='JPEG')

            img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('ascii')
            base64_images.append(f"data:image/jpeg;base64,{img_base64}")

    pdf_document.close()

    return base64_images, selected_spans


def visionpolicydetails(pdf_path, page_num, client, AZURE_OPENAI_CHATGPT_DEPLOYMENT, max_images=10):
    """
    Process the PDF through the pipeline.

    Args:
        pdf_path (str): Path to the PDF document
        page_num (list or tuple): Page number span(s)
        client: Azure OpenAI client instance
        AZURE_OPENAI_CHATGPT_DEPLOYMENT: Deployment name for Azure OpenAI
        max_images (int): Maximum number of images to process (default: 10)

    Returns:
        dict: Parsed policy information
    """
    try:
        # Extract base64 images and select spans
        base64_images, selected_spans = extract_and_select_page_images(pdf_path, page_num, max_items=2, max_images=max_images)
        print(f"Selected Page Spans: {selected_spans}")

        # Generate responses for selected images
        response = response_generator(client, AZURE_OPENAI_CHATGPT_DEPLOYMENT, base64_images)
        policy_info = parse_policy_information(response.choices[0].message.content, client, AZURE_OPENAI_CHATGPT_DEPLOYMENT)
        return policy_info

    except Exception as e:
        print(f"Error in processing pipeline: {str(e)}")
        return {}

if __name__ == '__main__':

    pass


    # pdfpath = input("Enter the path to the pdf path")

    # base64_image = extract_pages_as_base64_images(pdfpath)
    # response_received = response_generator(client, AZURE_OPENAI_CHATGPT_DEPLOYMENT, base64_image)
    # document_number = response_received.choices[0].message.content
    # policyinfodict = parse_policy_information(document_number, client, AZURE_OPENAI_CHATGPT_DEPLOYMENT)
    # print("Printing Document Number: ", policyinfodict)
    # print(type(policyinfodict['policy_start_date']))