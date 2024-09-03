# ocr the first 2 pages to text 
# will be used to determine if we need to process further

from anthropic import Anthropic
import os

from my_fncs import transcribe_pdf

# parameters
# ---

dname_in = "../../data_raw/documents/1983/"
dname_out = "../../results/norag_2_claude_api/texts/"

fnames = [
    # "ATCM12_wp014_rev1_e.pdf",
    # "ATCM12_wp015_e.pdf",
    # "ATCM12_wp016_rev1_e.pdf",
    # "ATCM12_wp017_rev1_e.pdf",
    "ATCM12_wp018_e.pdf",
    "ATCM12_wp019_e.pdf",
    "ATCM12_wp020_e.pdf",
    "ATCM12_wp021_e.pdf",
    "ATCM12_wp022_e.pdf",
    "ATCM12_wp023_e.pdf",
    "ATCM12_wp024_rev1_e.pdf",
    "ATCM12_wp024_rev2_e.pdf",
    "ATCM12_wp025_e.pdf",
    "ATCM12_wp026_e.pdf",
    "ATCM12_wp027_rev1_e.pdf",
    "ATCM12_wp028_rev1_e.pdf",
]

# convert pdfs to text and save
# ---

# initialize Anthropic client
client = Anthropic(
    # This is a placeholder. Replace with your actual API key
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

for fname in fnames:
    print(f"Processing file {fname}")
    pdf_path = os.path.join(dname_in, fname)

    # Use the function
    transcribed_text = transcribe_pdf(pdf_path, client, 1, 2)

    # Save the transcribed text to a file
    fname_out = fname.split(".pdf")[0] + "_first_two_pages.txt"
    output_name = os.path.join(dname_out, fname_out)
    with open(output_name, "w", encoding="utf-8") as output_file:
        output_file.write(transcribed_text)

    print(f" Transcription complete. Output saved to {output_name}")