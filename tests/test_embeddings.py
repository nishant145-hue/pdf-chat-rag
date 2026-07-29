from app.services.pdf_reader import extract_text_from_pdf
from app.utils.helpers import clean_text
from app.services.chunker import split_text
from app.services.embeddings import create_embeddings


def main():

    text = extract_text_from_pdf("uploads/SPPU ONLINE RESULT DISPLAY.pdf")

    text = clean_text(text)

    chunks = split_text(
        text,
        chunk_size=300,
        overlap=50
    )

    embeddings = create_embeddings(chunks)

    print(f"Total Chunks: {len(chunks)}")
    print(f"Total Embeddings: {len(embeddings)}")

    print("\nEmbedding Dimension:")
    print(len(embeddings[0]))

    print("\nFirst 10 values of the first embedding:")
    print(embeddings[0][:10])


if __name__ == "__main__":
    main()