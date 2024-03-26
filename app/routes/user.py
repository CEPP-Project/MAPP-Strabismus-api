from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.utils.auth.auth_utils import get_current_user
import app.db.models as models
import app.db.schemas as schemas

user_router = APIRouter(
    prefix='/user',
    tags=['User']
)

@user_router.get("/me")
async def get_my_user(current_user: schemas.UserInDB = Depends(get_current_user)):
    return current_user

@user_router.get("/history")
async def get_my_history(current_user: schemas.UserInDB = Depends(get_current_user), db: Session = Depends(get_db)):
    history = db.query(models.Historys).filter_by(user_id=current_user.user_id).order_by(models.Historys.timestamp.desc()).all()
    return history

@user_router.get("/graph")
async def get_my_graph(current_user: schemas.UserInDB = Depends(get_current_user), db: Session = Depends(get_db)):
    graph_data = db.query(models.Historys).filter_by(user_id=current_user.user_id).order_by(models.Historys.timestamp.desc()).limit(7).all()
    return graph_data