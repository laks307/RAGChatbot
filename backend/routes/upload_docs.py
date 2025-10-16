from logging import exception
from log import logger
from model.vectordb import embed_docs
from typing import List
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException


router = APIRouter()

@router.post('/upload-docs/')
async def upload_docs(files: List[UploadFile] = File(...)):
    try:
        logger.info("uploading docs, proceeding to vector db.py")
        embed_docs(files)
        logger.debug("upload docs route success")
        return {"messages": "Files processed and vectorstore updated"}

    except exception as e:
        logger.error(e)
        raise HTTPException(status_code=404, detail=str(e))