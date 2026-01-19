import re
import json
from pydantic import BaseModel, Field
from monads.monad_class import monad_wrapper


class PolicyDetails(BaseModel):
    start_date: str = Field(..., description="The start date of the policy in MM/DD/YYYY format.")
    end_date: str = Field(..., description="The end date of the policy in MM/DD/YYYY format.")
    policy_number: str = Field(..., description="A unique identifier for the policy.")
    holder_name: str = Field(..., description="Policy Holder's Name.")


def get_tools_and_choice():
    tools = [
        {
            "type": "function",
            "strict": True,
            "function": {
                "name": "PolicyDetails",
                "parameters": {
                    "properties": {
                        "start_date": {"description": "The start date of the policy in MM/DD/YYYY format.", "title": "Start Date", "type": "string"},
                        "end_date": {"description": "The end date of the policy in MM/DD/YYYY format.", "title": "End Date", "type": "string"},
                        "policy_number": {"description": "A unique identifier for the policy.", "title": "Policy Number", "type": "string"},
                        "holder_name": {"description": "Policy Holder's Name.", "title": "Holder Name", "type": "string"},
                    },
                    "required": ["start_date", "end_date", "policy_number", "holder_name"],
                    "title": "PolicyDetails",
                    "type": "object",
                    "additionalProperties": False,
                },
            },
        }
    ]

    tool_choice = {
        "type": "function",
        "function": {"name": "PolicyDetails"},
    }
    return tools, tool_choice


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
def extract_context_from_json(adobe_json: dict, index: tuple):
    # Open the PDF file
    # with open(json_file_path, 'rb') as file:
    # adobe_json = json.load(file)
    adobe_json = adobe_json["elements"]
    text_list = extract_text_from_adobe_json(filter_adobe_json_by_object_id(adobe_json, index[0], index[1]).unwrap()).unwrap()

    #print(text_list)
    
    finaltext = ""
    for text in text_list:
        text = preprocess_text(text).unwrap()
        finaltext = finaltext +"\n" + text
        # Find all matches in the preprocessed text
    return finaltext


@monad_wrapper
def normalize_pattern(s):
    # Use regular expressions to remove spaces and parentheses
    return re.sub(r"[ ()-]", "", s)


def response_generator(text, tools, tool_choice, client, AZURE_OPENAI_CHATGPT_DEPLOYMENT):
    response = client.chat.completions.create(
        model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        tools=tools,
        tool_choice=tool_choice,
        messages=[
            {
                "role": "system",
                "content": """
You are an expert insurance document analyzer. You will be provided with text extracted from an insurance declaration document. Your task is to:

1. Extract the Policy Holder's full name
2. Extract the Policy Start Date and convert it to US date format (MM/DD/YYYY)
3. Extract the Policy End Date and convert it to US date format (MM/DD/YYYY)

Rules for date conversion:
- If a date is in DD/MM/YYYY format, convert it to MM/DD/YYYY
- If a date is in YYYY-MM-DD format, convert it to MM/DD/YYYY
- If a date is already in MM/DD/YYYY format, keep it as is
- If a date contains month names (e.g., "15 January 2024" or "January 15, 2024"), convert to MM/DD/YYYY

Return ONLY a JSON object with exactly these three fields:
{
    "holder_name": "extracted name",
    "start_date": "MM/DD/YYYY",
    "end_date": "MM/DD/YYYY",
    "policy_number": "HO2JBGAOUBG03"
}

Important:
- Do not include any explanations or additional text
- If any field cannot be found, use null as its value
- Maintain exact spelling of the policyholder's name as found in the document
- For compound names, include all parts (first, middle, last, suffixes)
- If multiple potential policyholder names are found, use the one explicitly labeled as policyholder or primary insured
            """,
            },
            {"role": "user", "content": text},
        ],
    )
    return PolicyDetails.model_validate(json.loads(response.choices[0].message.tool_calls[0].function.arguments))


@monad_wrapper
def extractPolicyDetails(adobe_json, index, client, AZURE_OPENAI_CHATGPT_DEPLOYMENT):
    # Handle case where index is empty
    if not index:  # If index is an empty list or None
        # Extract text from the entire JSON
        entire_text = extract_context_from_json(adobe_json, (0, len(adobe_json["elements"]))).unwrap()
        tools, tool_choice = get_tools_and_choice()
        policy_details = response_generator(entire_text, tools, tool_choice, client, AZURE_OPENAI_CHATGPT_DEPLOYMENT)
        return policy_details.start_date, policy_details.end_date, policy_details.policy_number, policy_details.holder_name

    # If index is not empty, process each span normally
    extracted_text = []
    for start, end in index:
        # Validate indices
        if start < 0 or end < start:
            raise ValueError(f"Invalid span indices: ({start}, {end})")
                
        # Extract text from the current span
        span_text = extract_context_from_json(adobe_json, (start, end)).unwrap()
                    
        if span_text:
            extracted_text.append(span_text)

    # Concatenate all extracted text pieces with a space between them
    combined_text = " ".join(extracted_text)
    tools, tool_choice = get_tools_and_choice()
    policy_details = response_generator(combined_text, tools, tool_choice, client, AZURE_OPENAI_CHATGPT_DEPLOYMENT)
    return policy_details.start_date, policy_details.end_date, policy_details.policy_number, policy_details.holder_name