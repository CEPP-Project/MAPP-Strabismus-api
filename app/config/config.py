from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    DATABASE_URL: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    JWT_SECRET_KEY: str
    MODEL_L: str
    MODEL_M: str
    MODEL_R: str
    ML_PATH: str

    model_config = SettingsConfigDict(env_file='.env')

# @lru_cache
def get_settings():
    return Settings()

# def reload_settings():
#     get_settings.cache_clear()