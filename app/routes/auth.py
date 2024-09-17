from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.schemas import UserCreate
from app.db.database import get_db
from app.utils.auth.auth_utils import create_access_token, get_hashed_password, verify_password
import app.db.models as models
import app.db.schemas as schemas

auth_router = APIRouter(
    prefix='/auth',
    tags=['Auth']
)

# Right now, we don't need to register on prod
# @auth_router.post("/register")
# async def register(user: UserCreate, db: Session = Depends(get_db)):
#     existing_user = db.query(models.Users).filter_by(username=user.username).first()
#     if existing_user:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    
#     hashed_password = get_hashed_password(user.password)
#     new_user = models.Users(username=user.username, password=hashed_password)

#     db.add(new_user)
#     db.commit()
#     db.refresh(new_user)

#     return {"message": "User created successfully"}

@auth_router.post("/login", response_model=schemas.TokenSchema)
async def login(request: UserCreate, db: Session = Depends(get_db)):
    user = db.query(models.Users).filter_by(username=request.username).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect username or password")
    hashed_pass = user.password
    if not verify_password(request.password, hashed_pass):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect username or password"
        )
    
    access=create_access_token(user.user_id)
    return {
        "access_token": access,
    }