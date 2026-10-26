"""Browser UI for upload and query."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

_STATIC = Path(__file__).resolve().parent.parent / "static"

router = APIRouter(include_in_schema=False)


@router.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(_STATIC / "index.html")
