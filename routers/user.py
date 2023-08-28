from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from routers.auth import get_current_user
from sqlalchemy.orm import Session
from typing import Annotated
from database import SessionLocal
import models

router = APIRouter(
    prefix="/user",
    tags=["user"]
)

templates = Jinja2Templates(directory="templates")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]


@router.get("/", response_class=HTMLResponse)
async def get_user_page(user: user_dependency, db: db_dependency, request: Request):
    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    user_info = db.query(models.Users).filter(models.Users.id == user.get('id')).first()
    return templates.TemplateResponse("user.html",
                                      {"request": request, "user": user, "user_info": user_info})
