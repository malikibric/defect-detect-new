from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from core import security
from core.config import settings
from core.deps import get_current_active_user
from models.database import get_db
from models.db_models import User
from models.schemas import TokenResponse, UserPublic, UserSignupRequest

router = APIRouter()


@router.post("/login/access-token", response_model=TokenResponse)
async def login_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalars().first()
    
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password",
        )
    elif not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return TokenResponse(
        access_token=security.create_access_token(user.id, expires_delta=access_token_expires),
        token_type="bearer",
    )


@router.post("/signup", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def create_user_signup(
    payload: UserSignupRequest,
    db: AsyncSession = Depends(get_db),
) -> UserPublic:
    hashed_password = security.get_password_hash(payload.password)
    user = User(
        email=payload.email,
        hashed_password=hashed_password,
    )
    db.add(user)
    try:
        await db.commit()
        await db.refresh(user)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=400,
            detail="A user with this email already exists.",
        )

    return UserPublic(
        id=user.id,
        email=user.email,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
    )


@router.get("/me", response_model=UserPublic)
async def read_current_user(current_user: User = Depends(get_current_active_user)) -> UserPublic:
    return UserPublic(
        id=current_user.id,
        email=current_user.email,
        is_active=current_user.is_active,
        is_superuser=current_user.is_superuser,
    )

