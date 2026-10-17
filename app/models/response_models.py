
"""
Response Models

These models define the JSON responses returned by the API.
"""

from pydantic import BaseModel, Field


class MessageResponse(BaseModel):
    """Generic response message."""

    message: str = Field(
        ...,
        description="Success or information message.",
    )


class UploadResponse(BaseModel):
    """Response returned after uploading a PDF."""

    document_id: str = Field(
        ...,
        description="Unique document identifier.",
    )

    filename: str = Field(
        ...,
        description="Uploaded PDF filename.",
    )

    status: str = Field(
        default="ready",
        description="Document processing status.",
    )


class Citation(BaseModel):
    """Source passage supporting an answer."""

    source: str | None = Field(
        default=None,
        description="Source PDF filename.",
    )

    page: int | None = Field(
        default=None,
        description="One-based page number in the source PDF.",
    )

    text: str = Field(
        ...,
        description="Retrieved passage associated with the citation.",
    )


class AnswerResponse(BaseModel):
    """Response returned after asking a question."""

    answer: str = Field(
        ...,
        description="Answer generated from the uploaded document.",
    )

    citations: list[Citation] = Field(
        default_factory=list,
        description="Source passages used to generate the answer.",
    )


class StatusResponse(BaseModel):
    """Server status response."""

    documents_loaded: int = Field(
        ...,
        description="Number of PDFs currently loaded into memory.",
    )


class DocumentInfo(BaseModel):
    """Information about one uploaded document."""

    document_id: str
    filename: str


class DocumentsResponse(BaseModel):
    """Response for listing all uploaded documents."""

    documents: list[DocumentInfo]
