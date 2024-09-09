import os

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from fastrag.config import UPLOAD_DIRECTORY
from fastrag.database import get_db
from fastrag.models import File as FileModel

router = APIRouter()


@router.post("/upload/")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    db_file = FileModel(filename=file.filename, content_type=file.content_type, path=file_location)
    db.add(db_file)
    db.commit()
    db.refresh(db_file)

    return {"filename": file.filename, "content_type": file.content_type}
