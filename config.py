from pydantic import BaseModel
import os


class Settings(BaseModel):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    data_dir: str = os.getenv("DATA_DIR", "data")
    faiss_index_path: str = os.path.join(data_dir, "faiss.index")
    metadata_path: str = os.path.join(data_dir, "metadata.pkl")
    rate_limit_upload: str = "5/minute"
    rate_limit_query: str = "20/minute"


settings = Settings()
os.makedirs(settings.data_dir, exist_ok=True)
