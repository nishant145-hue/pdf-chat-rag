"""
Request Models

These models define the JSON format expected by the API.
"""

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """
    Request model for asking a question.

    Example
    -------
    {
        "document_id": "8f23d5e8-7e5a-49e8-bdf8-0b86b4b4d2a8",
        "question": "Explain variables in Python"
    }
    """

    document_id: str = Field(
        ...,
        description="Unique document ID returned by the upload endpoint."
    )

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question to ask about the uploaded PDF."
    )


class UploadRequest(BaseModel):
    """
    Optional model for future enhancements.

    Currently, the upload endpoint uses UploadFile.
    This class is kept for future metadata support.
    """

    filename: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Original PDF filename."
    )