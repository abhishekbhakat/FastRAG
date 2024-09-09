from typing import Annotated

from fastapi import UploadFile
from pydantic import BaseModel, Field


class UploadForm(BaseModel):
    file: Annotated[UploadFile, Field(description="Upload a file")]


class ChatForm(BaseModel):
    message: str = Field(title="Message")


class URLForm(BaseModel):
    url: str = Field(title="URL")


class MessageHistoryModel(BaseModel):
    message: str
