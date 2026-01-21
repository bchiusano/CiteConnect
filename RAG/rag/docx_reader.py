from __future__ import annotations
from pathlib import Path


# NOTE: This helper is currently not used anywhere in the active code paths.
# It is kept here for potential future use, but effectively disabled to make
# the dependency on python-docx optional from the core package perspective.
#
# If you want to use it in your own scripts, you can uncomment the function
# below or copy it into your own code.

# def read_docx(path: str | Path) -> str:
#     """
#     Read text from a .docx file and return it as a single string.
#     """
#     from docx import Document  # python-docx
#
#     p = Path(path)
#     doc = Document(str(p))
#     parts = [para.text for para in doc.paragraphs if para.text and para.text.strip()]
#     return "\n".join(parts).strip()
