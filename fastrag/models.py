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


class ServiceStatus(BaseModel):
    service_name: str
    status: str  # "online" or "offline"
    last_checked: str  # Timestamp of the last check

    def save(self):
        # Implement the save method to store the status in the database or other storage
        pass

    @classmethod
    def get(cls, service_name: str):
        # Implement the get method to retrieve the status from the database or other storage
        return cls(service_name=service_name, status="online", last_checked="2024-09-19T12:00:00")  # Placeholder
