from unittest.mock import patch

import pytest

from app.config import settings
from app.services.rag_pipeline import RAGPipeline


def create_test_pdf(pdf_path):
    """Create a minimal text-based PDF without extra dependencies."""

    lines = [
        "Project Aurora is an intelligent document search system.",
        "It uses semantic embeddings to find relevant information.",
        "The system stores document vectors in a FAISS index.",
        "Retrieved passages provide context for answer generation.",
    ]

    def escape_pdf_text(text):
        return (
            text.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )

    commands = ["BT", "/F1 12 Tf", "72 750 Td"]

    for index, line in enumerate(lines):
        if index:
            commands.append("0 -20 Td")

        commands.append(f"({escape_pdf_text(line)}) Tj")

    commands.append("ET")

    content = "\n".join(commands).encode("ascii")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R "
            b"/MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> "
            b"/Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            b"<< /Length "
            + str(len(content)).encode("ascii")
            + b" >>\nstream\n"
            + content
            + b"\nendstream"
        ),
    ]

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]

    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)

    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")

    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    pdf.extend(
        (
            f"trailer\n"
            f"<< /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n"
            f"%%EOF\n"
        ).encode("ascii")
    )

    pdf_path.write_bytes(pdf)


@pytest.fixture
def sample_pdf(tmp_path):
    """Generate a real PDF for integration testing."""

    pdf_path = tmp_path / "rag_integration_sample.pdf"
    create_test_pdf(pdf_path)

    return pdf_path


def test_real_pdf_pipeline_returns_citations(
    sample_pdf,
    monkeypatch,
):
    """Test PDF extraction, embeddings, FAISS retrieval, and citations."""

    monkeypatch.setattr(
        settings,
        "ENABLE_RERANKING",
        False,
    )

    pipeline = None

    try:
        with patch(
            "app.services.rag_pipeline.generate_answer",
            return_value=(
                "Project Aurora uses semantic embeddings "
                "and a FAISS index."
            ),
        ) as mock_generate_answer:

            # Exercise the real PDF processing and embedding pipeline.
            pipeline = RAGPipeline(str(sample_pdf))

            assert pipeline.total_chunks() > 0
            assert pipeline.vector_store.is_loaded()

            # Confirm the actual PDF text was extracted.
            assert any(
                "Project Aurora" in chunk
                for chunk in pipeline.vector_store.chunks
            )

            result = pipeline.ask_with_citations(
                "How does Project Aurora find information?"
            )

            assert result["answer"] == (
                "Project Aurora uses semantic embeddings "
                "and a FAISS index."
            )

            assert result["citations"]

            citation = result["citations"][0]

            assert citation["source"] == sample_pdf.name
            assert citation["page"] == 1
            assert citation["text"].strip()

            mock_generate_answer.assert_called_once()

    finally:
        # Remove only this test's persisted vector-store data.
        if pipeline is not None:
            pipeline.vector_store.delete(pipeline.filename)