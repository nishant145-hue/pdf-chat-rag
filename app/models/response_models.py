"""
Response Models

These models define the JSON responses returned by the API.
"""

from pydantic import BaseModel, Field


class MessageResponse(BaseModel):
    """
    Generic response message.
    """

    message: str = Field(
        ...,
        description="Success or information message."
    )


class UploadResponse(BaseModel):
    """
    Response returned after uploading a PDF.
    """

    document_id: str = Field(
        ...,
        description="Unique document identifier."
    )

    filename: str = Field(
        ...,
        description="Uploaded PDF filename."
    )

    status: str = Field(
        default="ready",
        description="Document processing status."
    )


class AnswerResponse(BaseModel):
    """
    Response returned after asking a question.
    """

    answer: str = Field(
        ...,
        description="Answer generated from the uploaded document."
    )


class StatusResponse(BaseModel):
    """
    Server status response.
    """

    documents_loaded: int = Field(
        ...,
        description="Number of PDFs currently loaded into memory."
    )


class DocumentInfo(BaseModel):
    """
    Information about one uploaded document.
    """

    document_id: str

    filename: str


class DocumentsResponse(BaseModel):
    """
    Response for listing all uploaded documents.
    """

    documents: list[DocumentInfo]