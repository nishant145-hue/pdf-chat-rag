from app.services.pdf_reader import extract_text_from_pdf
from app.utils.helpers import clean_text
from app.services.chunker import split_text
from app.services.embeddings import create_embeddings, create_embedding
from app.services.vector_store import VectorStore


def main():
    # Prepare the document
    text = extract_text_from_pdf("uploads/SPPU ONLINE RESULT DISPLAY.pdf")
    text = clean_text(text)
    chunks = split_text(text, chunk_size=300, overlap=50)

    # Create document embeddings
    embeddings = create_embeddings(chunks)

    # Build FAISS index
    store = VectorStore()
    store.build_index(embeddings, chunks)

    # User question
    question = "What is Database Management System?"

    # Create query embedding
    query_embedding = create_embedding(question)

    # Search
    results = store.search(query_embedding, k=3)

    print("\nTop 3 Matching Chunks:\n")

    for i, chunk in enumerate(results, start=1):
        print("=" * 60)
        print(f"Result {i}")
        print("=" * 60)
        print(chunk)
        print()
        

if __name__ == "__main__":
    main()