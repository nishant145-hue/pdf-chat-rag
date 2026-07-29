from app.services.pdf_reader import extract_text_from_pdf
from app.utils.helpers import clean_text
from app.services.chunker import split_text
from app.services.embeddings import create_embeddings, create_embedding
from app.services.vector_store import VectorStore
from app.services.llm import generate_answer


def main():

    print("Reading PDF...")
    text = extract_text_from_pdf("uploads/SPPU ONLINE RESULT DISPLAY.pdf")

    print("Cleaning text...")
    text = clean_text(text)

    print("Creating chunks...")
    chunks = split_text(
        text,
        chunk_size=300,
        overlap=50
    )

    print(f"Chunks Created: {len(chunks)}")

    print("Generating embeddings...")
    embeddings = create_embeddings(chunks)

    print("Building FAISS index...")
    store = VectorStore()
    store.build_index(embeddings, chunks)

    question = input("\nAsk your question: ")

    print("Creating query embedding...")
    query_embedding = create_embedding(question)

    print("Searching similar chunks...")
    retrieved_chunks = store.search(query_embedding, k=3)

    print("\nRetrieved Chunks\n")
    print("=" * 70)

    for i, chunk in enumerate(retrieved_chunks, start=1):
        print(f"\nChunk {i}\n")
        print(chunk)
        print("-" * 70)

    print("\nGenerating Answer...\n")

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)


if __name__ == "__main__":
    main()