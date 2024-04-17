from typing import Optional
from pydantic import BaseModel


class InputSchema(BaseModel):
    prompt: str
    image: Optional[str] = None
    input_dir: Optional[str] = None
    output_path: Optional[str] = None