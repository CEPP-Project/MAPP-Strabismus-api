from sqlalchemy import TIMESTAMP, JSON, String, UUID, Column, ForeignKey, text
from app.db.database import Base
import uuid

class Users(Base):
    __tablename__ = 'users'

    user_id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    username = Column(String, unique=True)
    password = Column(String)

class Historys(Base):
    __tablename__ = 'historys'

    result_id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    result = Column(JSON)
    user_id = Column(UUID, ForeignKey('users.user_id'))
    timestamp = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    