from typing import Annotated

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])
auth_service = AuthService()


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=1024)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=1024)


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    created_at: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(request: RegisterRequest):
    try:
        user = auth_service.register_user(
            email=str(request.email),
            password=request.password,
        )
        return UserResponse(**user)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest):
    user = auth_service.authenticate_user(
        email=str(request.email),
        password=request.password,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_service.create_access_token(user["id"])

    return TokenResponse(
        access_token=token,
        expires_in=auth_service_token_expiry(),
    )


def auth_service_token_expiry() -> int:
    from app.config import settings

    return settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60