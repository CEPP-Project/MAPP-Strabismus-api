from fastapi import Depends, HTTPException, status
from passlib.context import CryptContext
from typing import Union, Any
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from app.config.config import get_settings
from app.db.database import get_db
from app.utils.auth.auth_bearer import JWTBearer
import app.db.models as models
import app.db.schemas as schemas

setting = get_settings()
jwt_expire = setting.ACCESS_TOKEN_EXPIRE_MINUTES
alg = "HS256"
jwt_secret = setting.JWT_SECRET_KEY

password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_hashed_password(password: str) -> str:
    return password_context.hash(password)

def verify_password(password: str, hashed_pass: str) -> bool:
    return password_context.verify(password, hashed_pass)

def create_access_token(subject: Union[str, Any], expires_delta: int = None) -> str:
    if expires_delta is not None:
        expires_delta = datetime.now(timezone.utc) + expires_delta
        
    else:
        expires_delta = datetime.now(timezone.utc) + timedelta(minutes=jwt_expire)
         
    
    to_encode = {"exp": expires_delta, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, jwt_secret, alg)
     
    return encoded_jwt

async def get_current_user(token: str = Depends(JWTBearer()), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, jwt_secret, algorithms=alg)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = schemas.TokenData(user_id=user_id)
    except JWTError:
        raise credentials_exception
    
    user = db.query(models.Users).filter_by(user_id=token_data.user_id).first()
    if user is None:
        raise credentials_exception
    
    return user