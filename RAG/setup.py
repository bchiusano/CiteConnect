from setuptools import setup, find_packages
from pathlib import Path

# The package directory is RAG (uppercase) but package name is rag (lowercase)
# We need to explicitly map this
setup(
    name="rag",
    version="1.0.0",
    packages=["rag"],  # Explicitly specify the package
    package_dir={"rag": "."},  # Map 'rag' package to current directory (RAG/)
    install_requires=[
        "flagembedding",
        "sentence-transformers",
        "spacy",
        "pandas",
        "openpyxl",
        "psycopg2-binary",
        "sqlalchemy",
        "python-docx",
    ],
    python_requires=">=3.8",
)
