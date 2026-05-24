"""Build Pinecone metadata filters from routing/router hints."""

from __future__ import annotations


def build_pinecone_filter(source_files: list[str] | None = None) -> dict | None:
    """
    Build a Pinecone metadata filter dict.
    
    Args:
        source_files: Optional list of source PDF filenames to restrict search to.
        
    Returns:
        dict | None: Pinecone-compliant metadata filter dictionary, or None.
    """
    if not source_files:
        return None

    # Clean the input list
    source_files = [f for f in source_files if f]
    if not source_files:
        return None

    # If single file, use direct equality. If multiple, use $in operator.
    if len(source_files) == 1:
        return {"source_file": source_files[0]}

    return {"source_file": {"$in": source_files}}
