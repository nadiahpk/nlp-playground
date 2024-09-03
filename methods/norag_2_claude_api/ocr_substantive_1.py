# once we have checked the first two pages are relevant,
# ocr the remainder of the pdf

import pandas as pd
from anthropic import Anthropic
import os

from my_fncs import transcribe_pdf, load_document, write_to_file


# parameters
# ---

# directories for raw and processed data
dname_in_raw = "../../data_raw/documents/1983/"
dname_in_processed = "../../results/norag_2_claude_api/texts/"

# a spreadsheet summarising which Working Papers
# have substantive content
fname_in = "wp_substantive.csv"

# directory to save text from pdfs
dname_out = "../../results/norag_2_claude_api/texts/"


# get list of pdfs to ocr
# ---

df = pd.read_csv(os.path.join(dname_in_processed, fname_in))

# only process substantive files that are NOT drafts of Final Report
fname_txts = df[df["is_substantive"] & (~ df["is_final_report_draft"])]["file_name"]
fnames = [s.split("_first_two_pages.txt")[0] for s in fname_txts]


# convert pdfs to text and save
# ---

# initialize Anthropic client
client = Anthropic(
    # This is a placeholder. Replace with your actual API key
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

# fname = fnames[1]
for fname in fnames[2:]:
    print(f"Processing file {fname}")
    # get the first two pages in plain text
    fname_first_two = fname + "_first_two_pages.txt"
    first_two_pages = load_document(os.path.join(dname_in_processed, fname_first_two))
    
    # get pages 3 onwards
    pdf_path = os.path.join(dname_in_raw, fname + ".pdf")
    remainder = transcribe_pdf(pdf_path, client, 3)

    # join the first two pages to the remainder
    transcribed_text = first_two_pages + "\n\n" + remainder

    # save the transcribed text to a file
    fname_out = fname + ".txt"
    output_name = os.path.join(dname_out, fname_out)
    write_to_file(output_name, transcribed_text)

    print(f" Transcription complete. Output saved to {output_name}")