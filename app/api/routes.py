from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from app.models.request_models import QuestionRequest

from app.models.response_models import (
    AnswerResponse,
    DocumentsResponse,
    DocumentInfo,
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
async def upload_pdf(
    file: UploadFile = File(...)
):
    """
    Upload a PDF and build a new RAG pipeline.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is missing."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    upload_dir = Path("uploads")

    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / file.filename

    with open(file_path, "wb") as pdf_file:
        pdf_file.write(await file.read())

    document_id = manager.add_document(
        pdf_path=str(file_path),
        filename=file.filename,
    )

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        status="ready",
    )


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
            detail="Document not found."
        )

    answer = pipeline.ask(request.question)

    return AnswerResponse(
        answer=answer
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