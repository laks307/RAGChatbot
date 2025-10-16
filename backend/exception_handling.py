from fastapi import HTTPException, Request
from fastapi.openapi.utils import status_code_ranges
from fastapi.responses import JSONResponse
from log import logger

async def error_handling(request: Request, call_next):
    try:
        logger.debug("calling next request")
        return await call_next(request)
    except Exception as e:
        logger.exception("middleware exception handling Request failed", e)
        return JSONResponse(status_code=500, content={"error": str(e)})