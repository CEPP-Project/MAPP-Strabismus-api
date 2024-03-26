from pydantic import BaseModel, ConfigDict, Field
from uuid import UUID
from typing import List
from datetime import datetime

class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    model_config = ConfigDict(from_attributes=True)

    user_id: UUID = Field(..., alias='id')

class HistoryBase(BaseModel):
    result: List
    user_id: UUID

class HistoryCreate(HistoryBase):
    pass

class HistoryInDB(HistoryBase):
    model_config = ConfigDict(from_attributes=True)

    result_id: UUID = Field(..., alias='id')
    timestamp: datetime

class TokenSchema(BaseModel):
    access_token: str

class TokenData(BaseModel):
    user_id: UUID | None = None
