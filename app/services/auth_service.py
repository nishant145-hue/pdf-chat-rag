import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from pwdlib import PasswordHash

from app.config import settings


password_hash = PasswordHash.recommended()

DUMMY_PASSWORD_HASH = password_hash.hash(
    "dummy-password-for-timing-protection"
)


class AuthService:
    """Manage persistent users, password verification, and JWT tokens."""

    def __init__(self, database_path: Path | str | None = None):
        self.database_path = Path(
            database_path or settings.AUTH_DATABASE_PATH
        ).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_database()

    def _connect(self):
        connection = sqlite3.connect(
            str(self.database_path),
            timeout=10,
        )
        connection.row_factory = sqlite3.Row
        return connection


    def initialize_database(self):
        """Create authentication and document ownership tables."""
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    owner_user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (owner_user_id)
                        REFERENCES users(id)
                        ON DELETE CASCADE
                )
                """
            )



    @staticmethod
    def normalize_email(email: str) -> str:
        return email.strip().lower()

    def register_user(self, email: str, password: str) -> dict:
        """Register a user or raise ValueError for a duplicate email."""
        normalized_email = self.normalize_email(email)

        if not normalized_email:
            raise ValueError("Email is required.")

        if len(normalized_email) > 254:
            raise ValueError("Email is too long.")

        if len(password) < 12:
            raise ValueError(
                "Password must contain at least 12 characters."
            )

        if len(password) > 1024:
            raise ValueError("Password is too long.")

        hashed_password = password_hash.hash(password)
        created_at = datetime.now(timezone.utc).isoformat()

        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO users (email, password_hash, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (normalized_email, hashed_password, created_at),
                )
                user_id = cursor.lastrowid
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "An account with this email already exists."
            ) from exc

        return {
            "id": user_id,
            "email": normalized_email,
            "created_at": created_at,
        }

    def authenticate_user(
        self,
        email: str,
        password: str,
    ) -> dict | None:
        """Verify credentials without revealing which field was incorrect."""
        normalized_email = self.normalize_email(email)

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, email, password_hash, created_at
                FROM users
                WHERE email = ?
                """,
                (normalized_email,),
            ).fetchone()

        if row is None:
            password_hash.verify(password, DUMMY_PASSWORD_HASH)
            return None

        if not password_hash.verify(password, row["password_hash"]):
            return None

        return {
            "id": row["id"],
            "email": row["email"],
            "created_at": row["created_at"],
        }

    def create_access_token(self, user_id: int) -> str:
        """Create a short-lived JWT access token."""
        now = datetime.now(timezone.utc)

        payload = {
            "sub": str(user_id),
            "iat": now,
            "exp": now + timedelta(
                minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
            ),
        }

        return jwt.encode(
            payload,
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )

    def get_user_by_id(self, user_id: int) -> dict | None:
        """Return a public user record without its password hash."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, email, created_at
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()

        if row is None:
            return None

        return dict(row)


    def assign_document_owner(
        self,
        document_id: str,
        owner_user_id: int,
    ) -> None:
        """Associate a document with an existing user."""
        created_at = datetime.now(timezone.utc).isoformat()

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO documents (
                        document_id,
                        owner_user_id,
                        created_at
                    )
                    VALUES (?, ?, ?)
                    """,
                    (document_id, owner_user_id, created_at),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                "Document already exists or user does not exist."
            ) from exc

    def user_owns_document(
        self,
        document_id: str,
        owner_user_id: int,
    ) -> bool:
        """Return True only when the specified user owns the document."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM documents
                WHERE document_id = ?
                  AND owner_user_id = ?
                """,
                (document_id, owner_user_id),
            ).fetchone()

        return row is not None


    def get_document_owner(self, document_id: str) -> int | None:
        """Return a document's owner ID, or None if it has no owner."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT owner_user_id
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

        return row["owner_user_id"] if row is not None else None

    def delete_document_owner(self, document_id: str) -> bool:
        """Delete a document's ownership record.

        Returns True if a record was deleted, otherwise False.
        """
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            )

        return cursor.rowcount > 0


    def decode_access_token(self, token: str) -> int | None:
        """Validate a JWT and return its user ID."""
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
                options={"require": ["sub", "iat", "exp"]},
            )

            subject = payload.get("sub")

            if not isinstance(subject, str) or not subject.isdigit():
                return None

            user_id = int(subject)

            if user_id <= 0:
                return None

            return user_id

        except (jwt.InvalidTokenError, TypeError, ValueError):
            return None