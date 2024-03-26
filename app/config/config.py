from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    DATABASE_URL: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    JWT_SECRET_KEY: str
    MODEL_L_PATH: str
    MODEL_M_PATH: str
    MODEL_R_PATH: str
    PREDICT_MODEL_PATH: str
    PREDICT_DROP_MODEL_PATH: str

    model_config = SettingsConfigDict(env_file=".env")

# @lru_cache
def get_settings():
    return Settings()

def reload_settings():
    get_settings.cache_clear()