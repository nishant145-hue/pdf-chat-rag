from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from app.models.request_models import QuestionRequest
from app.models.response_models import (
    AnswerResponse,
    DocumentInfo,
    DocumentsResponse,
    MessageResponse,
    StatusResponse,
    UploadResponse,
)
from app.services.document_manager import DocumentManager

router = APIRouter()

# Global document manager
manager = DocumentManager()


@router.get("/")
def home():
    """
    Health check endpoint.
    """

    return {
        "message": "PDF Chat RAG API is running."
    }


@router.post(
    "/upload",
    response_model=UploadResponse,
)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF without overwriting existing documents."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is missing.",
        )

    original_filename = Path(file.filename.replace("\\", "/")).name

    if (
        not original_filename
        or original_filename in {".", ".."}
        or not original_filename.lower().endswith(".pdf")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    stored_filename = f"{uuid4()}_{original_filename}"
    file_path = upload_dir / stored_filename

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded PDF is empty.",
            )

        with open(file_path, "xb") as pdf_file:
            pdf_file.write(content)

        document_id = manager.add_document(
            pdf_path=str(file_path),
            filename=original_filename,
        )

        return UploadResponse(
            document_id=document_id,
            filename=original_filename,
            status="ready",
        )

    except HTTPException:
        if file_path.exists():
            file_path.unlink()
        raise

    except Exception as exc:
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail="Failed to process the uploaded PDF.",
        ) from exc

@router.get(
    "/status",
    response_model=StatusResponse,
)
def status():
    """
    Returns how many documents are loaded.
    """

    return StatusResponse(
        documents_loaded=manager.total_documents()
    )

@router.post(
    "/ask",
    response_model=AnswerResponse,
)
def ask_question(request: QuestionRequest):
    """
    Ask a question about a specific uploaded PDF.
    """

    pipeline = manager.get_pipeline(request.document_id)

    if pipeline is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    result = pipeline.ask_with_citations(request.question)

    return AnswerResponse(
        answer=result["answer"],
        citations=result["citations"],
    )



@router.get(
    "/documents",
    response_model=DocumentsResponse,
)
def list_documents():
    """
    Returns all uploaded documents.
    """

    documents = []

    for document in manager.list_documents():

        documents.append(
            DocumentInfo(
                document_id=document["document_id"],
                filename=document["filename"],
            )
        )

    return DocumentsResponse(
        documents=documents
    )


@router.delete(
    "/documents/{document_id}",
    response_model=MessageResponse,
)
def delete_document(document_id: str):
    """
    Delete an uploaded PDF.
    """

    deleted = manager.delete_document(document_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    return MessageResponse(
        message="Document deleted successfully."
    )
