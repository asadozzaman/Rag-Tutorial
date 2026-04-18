# Chunking Notes

## Why recursive chunking is a strong default

Recursive chunking tries to preserve natural boundaries such as paragraphs,
line breaks, and sentences before falling back to spaces or characters.
That usually creates chunks that are easier for retrieval systems to use.

## Why overlap matters

Overlap keeps neighboring context connected. If an answer begins near the end
of one chunk and continues into the next, overlap reduces the chance that the
retriever misses important information.

## Metadata to preserve

For production RAG systems, each chunk should carry metadata like source,
source type, page number, row number, URL, and section name.
This helps with filtering, debugging, observability, and user-facing citations.

## Common chunking failure

Very large chunks dilute relevance because they mix too many topics.
Very small chunks can lose context and break explanations into fragments.
