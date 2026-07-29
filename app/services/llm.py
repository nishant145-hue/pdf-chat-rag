import os

from dotenv import load_dotenv
from groq import Groq

from app.config import settings
from app.logger import logger

load_dotenv()

client = Groq(
    api_key=settings.GROQ_API_KEY,
)


def build_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build a structured RAG prompt.
    """

    return f"""
You are an expert AI assistant for question answering.

Your job is to answer ONLY using the supplied document context.

Rules:

1. Use ONLY the information present in the context.

2. Do NOT invent facts.

3. Do NOT assume missing information.

4. If the answer is not present in the context,
    reply exactly:

    "I couldn't find that in the uploaded document."

5. Keep the answer concise and accurate.

6. If appropriate, answer using bullet points.

7. If the context contains code,
    preserve formatting.

----------------------------------------
DOCUMENT CONTEXT
----------------------------------------

{context}

----------------------------------------
USER QUESTION
----------------------------------------

{question}

----------------------------------------
ANSWER
----------------------------------------
"""


def generate_answer(
    question: str,
    context: str,
) -> str:
    """
    Generate an answer using the Groq LLM.
    """

    prompt = build_prompt(
        question=question,
        context=context,
    )

    logger.info(
        "Sending request to Groq model: %s",
        settings.GROQ_MODEL,
    )

    try:

        response = client.chat.completions.create(

            model=settings.GROQ_MODEL,

            temperature=settings.TEMPERATURE,

            max_completion_tokens=settings.MAX_TOKENS,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers "
                        "questions strictly from the provided context."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        answer = response.choices[0].message.content

        logger.info("Groq response received successfully.")

        return answer.strip()

    except Exception:

        logger.exception(
            "Groq request failed."
        )

        raise