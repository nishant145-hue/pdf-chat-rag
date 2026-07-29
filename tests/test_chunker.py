from app.services.pdf_reader import extract_text_from_pdf
from app.utils.helpers import clean_text
from app.services.chunker import split_text


def main():
    print("Inside main")

    text = extract_text_from_pdf("uploads/SPPU ONLINE RESULT DISPLAY.pdf")
    text = clean_text(text)

    chunks = split_text(
        text,
        chunk_size=300,
        overlap=50
    )

    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print("=" * 50)
        print(f"Chunk {i}")
        print(chunk)


if __name__ == "__main__":
    main()