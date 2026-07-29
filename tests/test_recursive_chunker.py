from app.services.pdf_reader import extract_text_from_pdf
from app.utils.helpers import clean_text
from app.services.recursive_chunker import recursive_chunk


def main():

    text = extract_text_from_pdf(
        "uploads/SPPU ONLINE RESULT DISPLAY.pdf"
    )

    text = clean_text(text)

    chunks = recursive_chunk(
        text,
        chunk_size=300
    )

    print(f"\nTotal Chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks, start=1):

        print("=" * 60)
        print(f"Chunk {i}")
        print("=" * 60)
        print(chunk)
        print()


if __name__ == "__main__":
    main()