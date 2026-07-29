from app.services.rag_pipeline import RAGPipeline


def main():

    pipeline = RAGPipeline(
        "uploads/pythonlearn.pdf"
    )

    while True:

        question = input("\nAsk a question (or type 'exit'): ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        answer = pipeline.ask(question)

        print("\nAnswer:\n")
        print(answer)
        print("-" * 80)


if __name__ == "__main__":
    main()