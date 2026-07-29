from app.services.pdf_reader import extract_text_from_pdf
from app.utils.helpers import clean_text


def main():
    pdf_path = r"C:\Users\nisha\OneDrive\Desktop\PDF Chat RAG APP\uploads\SPPU ONLINE RESULT DISPLAY.pdf"

    try:
        raw_text = extract_text_from_pdf(pdf_path)

        cleaned_text = clean_text(raw_text)

        print("=" * 60)
        print("CLEANED PDF TEXT")
        print("=" * 60)
        print(cleaned_text)

    except FileNotFoundError as error:
        print(f"File Error: {error}")

    except ValueError as error:
        print(f"Value Error: {error}")


if __name__ == "__main__":
    main()