from fastapi import FastAPI
from app.routes.uploads import uploads_router

app = FastAPI()

app.include_router(uploads_router)