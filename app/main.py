from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.database import engine
import app.db.models as models
from app.routes.uploads import uploads_router
from app.routes.user import user_router
from app.routes.auth import auth_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['GET', 'POST'],
    )

models.Base.metadata.create_all(bind=engine)

app.include_router(uploads_router)
app.include_router(user_router)
app.include_router(auth_router)