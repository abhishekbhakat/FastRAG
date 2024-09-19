from typing import Annotated

from fastapi import UploadFile
from fastui.forms import FormFile
from pydantic import BaseModel, Field


class UploadForm(BaseModel):
    file: Annotated[UploadFile, FormFile(accept="*/*")] = Field(title="File")


class ChatForm(BaseModel):
    message: str = Field(title="Message")


class URLForm(BaseModel):
    url: str = Field(title="URL", min_length=1)


class MessageHistoryModel(BaseModel):
    message: str
