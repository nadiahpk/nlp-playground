
import base64
import io
from pdf2image import convert_from_path
import xml.etree.ElementTree as ET

# structure for multiple documents in prompt
def documents_to_xml_string(document_dictionary):
    """

    Overview:
    ---

    Anthropic recommend wrapping documents in long-context prompts in XML tags
    https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips

    The function below turns a dictionary with structure like:

        document_dictionary = {"doc_1.csv": content_from_doc_1, "doc_2.txt": ...}

    into an xml string with a structure like:

        <documents>
        <document index="1">
            <source>doc_1.csv</source>
            <document_content>
            {content_from_doc_1}
            </document_content>
        </document>
        <document index="2">
            <source>doc_2.txt</source>
            ...
        </document>
        ...
        </documents>
    

    Inputs:
    ---

    document_dictionary, Dict[str, str]
        Keys are potentially-fake but informative file names, 
        and values are the text content of file.


    Outputs:
    ---

    xml_string, str
        A string of the document content wrapped in XML tags.



    """

    # build the xml tree
    root = ET.Element("documents")
    
    for index, (source, content) in enumerate(document_dictionary.items(), start=1):
        doc = ET.SubElement(root, "document", index=str(index))
        ET.SubElement(doc, "source").text = source
        doc_content = ET.SubElement(doc, "document_content")
        doc_content.text = content
    
    # turn the xml into a string
    xml_string = ET.tostring(root, encoding='unicode', method='xml')

    return xml_string



# reading and writing plain text
# ---

def write_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)

def load_document(file_path: str) -> str:
    """Load document content from a file."""
    try:
        with open(file_path, "r") as file:
            return file.read()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

# ocr pdfs
# ---

def get_base64_encoded_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def transcribe_pdf(
    pdf_path, 
    anthropic_client,
    first_page = None, 
    last_page = None,
    anthropic_model_name = "claude-3-opus-20240229"
):
    """
    Use the Anthropic API to turn scanned pdfs into plain text.
    
    Inputs
    ---

    pdf_path, str
        Location of the pdf file

    anthropic_client
        e.g., client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    first_page, [int]
        Default is Page 1.

    last_page, [int]
        Default is last page of pdf.

    anthropic_model_name, [str]
        Default is "claude-3-opus-20240229"

    Outputs
    ---

    all_text, str
        All text in page range of pdf.
    """

    if first_page is None:
        first_page = 1

    images = convert_from_path(pdf_path, first_page=first_page, last_page=last_page)
    
    all_text = []
    
    # Process each page
    for i, image in enumerate(images):
        print(f" Processing page {first_page + i} of {first_page}--{first_page + len(images) - 1}")
        
        # Encode the image
        base64_image = get_base64_encoded_image(image)
        
        # Create the message for Claude
        message_list = [
            {
                "role": 'user',
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image}},
                    {"type": "text", "text": "Transcribe all the text from this image. Maintain the structure and formatting as much as possible. If the page contains a figure, make a note about it in square brackets."}
                ]
            }
        ]
        
        # Get the response from Claude
        response = anthropic_client.messages.create(
            model=anthropic_model_name,
            max_tokens=4096,  # Adjust as needed
            messages=message_list
        )
        
        # Append the transcribed text
        all_text.append(f"\n\n--- Scanned Page {first_page + i} ---\n\n{response.content[0].text}")
    
    return "\n".join(all_text)