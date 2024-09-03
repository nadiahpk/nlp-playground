# split minutes text into agenda items

import os
import re
from my_fncs import load_document, write_to_file

# parameters
# ---

# where is the final-report minutes?
dname_in = "../../results/norag_2_claude_api/texts/"
fname_in = "ATCM12_fr001_e_minutes.txt"

# where do you want to save the agenda items to?
dname_out = dname_in


# split into agenda items and save in separate files
# ---

text = load_document(os.path.join(dname_in, fname_in))
paragraphs = text.split("\n\n")

# chunks not items bc may be more than 1 agenda item
# per chunk
agenda_chunks = []

# find the first agenda item
iter_par = iter(paragraphs)
paragraph = next(iter_par)
while not re.match(r"^--- New Agenda Item ---", paragraph):
    paragraph = next(iter_par)

# append each agenda item into its own text
current_agenda_chunk = ""
while (paragraph := next(iter_par, None)) is not None:

    # ignore paragraphs that are the strings identifying the scanned page number
    # e.g., "\n--- Scanned Page 23 ---"
    if re.search(r'---\s*Scanned Page\s+(\d+)\s*---', paragraph):
        # I know this is safe to do bc I'm the one who added these markers
        continue

    # if this paragraph starts with my marker "--- New Agenda Item ---"
    # put it in a new chunk; otherwise, append whatever it is
    # to the previous chunk
    if re.match(r"^--- New Agenda Item ---", paragraph):
        # if this is a new agenda item, start a new chunk
        agenda_chunks.append(current_agenda_chunk)
        current_agenda_chunk = ""
    else:
        # append to the previous chunk
        current_agenda_chunk += "\n\n" + paragraph

# append the final paragraph
agenda_chunks.append(current_agenda_chunk)

# write to files
# ---

for i, chunk in enumerate(agenda_chunks):
    fname_out = f"{fname_in.split('.')[0]}_chunk_{i}.txt"
    file_path = os.path.join(dname_out, fname_out)
    write_to_file(file_path, chunk)