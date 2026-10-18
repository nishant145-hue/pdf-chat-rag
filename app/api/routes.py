from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
)

from app.api.auth_routes import auth_service
from app.api.dependencies import get_current_user
from app.config import settings
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
from app.services.pdf_reader import validate_pdf_file

router = APIRouter()

# Global document manager.
manager = DocumentManager()


def _require_owned_document(
    document_id: str,
    user_id: int,
) -> dict:
    """Return a document only when the requesting user owns it."""

    document = manager.get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    if not auth_service.user_owns_document(document_id, user_id):
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return document


@router.get("/")
def home():
    """Public API health information."""
    return {
        "message": "PDF Chat RAG API is running."
    }


@router.post(
    "/upload",
    response_model=UploadResponse,
)
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload a PDF and assign it to the authenticated user."""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is missing.",
        )

    original_filename = Path(
        file.filename.replace("\\", "/")
    ).name

    if (
        not original_filename
        or original_filename in {".", ".."}
        or not original_filename.lower().endswith(".pdf")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    upload_dir = settings.UPLOAD_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)

    stored_filename = f"{uuid4()}_{original_filename}"
    file_path = upload_dir / stored_filename

    document_id = None

    try:
        total_bytes = 0
        chunk_size = 1024 * 1024

        with open(file_path, "xb") as pdf_file:
            while True:
                chunk = await file.read(chunk_size)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > settings.MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Uploaded PDF exceeds the maximum "
                            "allowed size."
                        ),
                    )

                pdf_file.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded PDF is empty.",
            )

        try:
            validate_pdf_file(str(file_path))
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        document_id = manager.add_document(
            pdf_path=str(file_path),
            filename=original_filename,
            owner_user_id=current_user["id"],
        )

        # Persist ownership before reporting upload success.
        auth_service.assign_document_owner(
            document_id=document_id,
            owner_user_id=current_user["id"],
        )

        return UploadResponse(
            document_id=document_id,
            filename=original_filename,
            status="ready",
        )

    except HTTPException:
        if document_id is not None:
            manager.delete_document(document_id)
        elif file_path.exists():
            file_path.unlink()
        raise

    except Exception as exc:
        if document_id is not None:
            manager.delete_document(document_id)
        elif file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail="Failed to process the uploaded PDF.",
        ) from exc


@router.get(
    "/status",
    response_model=StatusResponse,
)
def status(
    current_user: dict = Depends(get_current_user),
):
    """Return the number of documents owned by the current user."""

    owned_documents = [
        document
        for document in manager.list_documents()
        if auth_service.user_owns_document(
            document["document_id"],
            current_user["id"],
        )
    ]

    return StatusResponse(
        documents_loaded=len(owned_documents)
    )


@router.post(
    "/ask",
    response_model=AnswerResponse,
)
def ask_question(
    request: QuestionRequest,
    current_user: dict = Depends(get_current_user),
):
    """Answer questions only about documents owned by the user."""

    document = _require_owned_document(
        request.document_id,
        current_user["id"],
    )

    pipeline = document["pipeline"]
    result = pipeline.ask_with_citations(request.question)

    return AnswerResponse(
        answer=result["answer"],
        citations=result["citations"],
    )


@router.get(
    "/documents",
    response_model=DocumentsResponse,
)
def list_documents(
    current_user: dict = Depends(get_current_user),
):
    """List only documents owned by the current user."""

    documents = []

    for document in manager.list_documents():
        if not auth_service.user_owns_document(
            document["document_id"],
            current_user["id"],
        ):
            continue

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
def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a document only when the current user owns it."""

    _require_owned_document(
        document_id,
        current_user["id"],
    )

    deleted = manager.delete_document(document_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    auth_service.delete_document_owner(document_id)

    return MessageResponse(
        message="Document deleted successfully."
)