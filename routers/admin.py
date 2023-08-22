from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from typing import Annotated
from models import RtspUrls, Users
from database import SessionLocal
from routers.auth import get_current_user
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    responses={401: {"user": "Not authorized"}}
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
async def admin(user: user_dependency, db: db_dependency, request: Request):
    if user is None or user.get('role') != 'admin':
        raise HTTPException(status_code=401, detail='Authentication Failed')
    admins = db.query(Users).filter_by(role='admin').all()
    operators = db.query(Users).filter_by(role='operator').all()
    pending_operators = db.query(Users).filter_by(role='operator', approved=False).all()
    pending_operators = [operator.name for operator in pending_operators]
    urls = db.query(RtspUrls.id, RtspUrls.url).all()
    return templates.TemplateResponse("admin.html",
                                      {"request": request, "admins": admins, "operators": operators,
                                       "pending_operators": pending_operators, "urls": urls, "user": user})


@router.get("/add_rtsp", response_class=HTMLResponse)
async def show_add_rtsp_form(request: Request, user: user_dependency):
    if user is None or user.get('role') != 'admin':
        raise HTTPException(status_code=401, detail='Authentication Failed')

    return templates.TemplateResponse("add_rtsp.html", {"request": request, "user": user})


@router.post("/add_rtsp", response_class=HTMLResponse)
async def process_add_rtsp_form(request: Request, user: user_dependency, db: db_dependency):
    if user is None or user.get('role') != 'admin':
        raise HTTPException(status_code=401, detail='Authentication Failed')

    form_data = await request.form()
    url = form_data.get("url")

    if not url:
        return templates.TemplateResponse("add_rtsp.html", {"request": request, "user": user})

    validation = db.query(RtspUrls).filter(RtspUrls.url == url).first()
    if validation is not None:
        msg = "Bu url zaten kayıtlı"
        return templates.TemplateResponse("add_rtsp.html", {"request": request, "msg": msg, "user": user})

    url_model = RtspUrls()
    url_model.url = url
    db.add(url_model)
    db.commit()

    return RedirectResponse(url="/admin", status_code=303)


@router.post("/remove_rtsp/{url_id}")
async def remove_rtsp(url_id: int, user: user_dependency, db: db_dependency):
    if user is None or user.get('role') != 'admin':
        raise HTTPException(status_code=401, detail='Authentication Failed')
    rtsp_to_delete = db.query(RtspUrls).filter_by(id=url_id).first()
    if rtsp_to_delete:
        db.delete(rtsp_to_delete)
        db.commit()
        return RedirectResponse(url="/admin", status_code=303)
    else:
        raise HTTPException(status_code=404, detail='url not found')


@router.post("/admin_role/{user_id}")
async def admin_role(user_id: int, user: user_dependency, db: db_dependency):
    if user is None or user.get('role') != 'admin':
        raise HTTPException(status_code=401, detail='Authentication Failed')

    user_to_promote = db.query(Users).filter_by(id=user_id, role='operator').first()
    if user_to_promote:
        user_to_promote.role = 'admin'
        db.commit()
        return RedirectResponse(url="/admin", status_code=303)
    else:
        raise HTTPException(status_code=404, detail='User not found')


@router.post("/remove_operator/{user_id}")
async def remove_operator(user: user_dependency, db: db_dependency, user_id: int):
    if user is None or user.get('role') != 'admin':
        raise HTTPException(status_code=401, detail='Authentication Failed')

    user_to_remove = db.query(Users).filter_by(id=user_id, role='operator').first()
    if user_to_remove:
        db.delete(user_to_remove)
        db.commit()
        return RedirectResponse(url="/admin", status_code=303)
    else:
        raise HTTPException(status_code=404, detail='User not found')


@router.post("/approve_operator/{user_id}")
async def approve_operator(user_id: int, user: user_dependency, db: db_dependency):
    if user is None or user.get('role') != 'admin':
        raise HTTPException(status_code=401, detail='Authentication Failed')

    user_to_approve = db.query(Users).filter_by(id=user_id, role='operator', approved=False).first()
    if user_to_approve:
        user_to_approve.approved = True
        db.commit()
        return RedirectResponse(url="/admin", status_code=303)
    else:
        raise HTTPException(status_code=404, detail='User not found')


@router.post("/reject_operator/{user_id}")
async def reject_operator(user_id: int, user: user_dependency, db: db_dependency):
    if user is None or user.get('role') != 'admin':
        raise HTTPException(status_code=401, detail='Authentication Failed')

    user_to_reject = db.query(Users).filter_by(id=user_id, role='operator', approved=False).first()
    if user_to_reject:
        db.delete(user_to_reject)
        db.commit()
        return RedirectResponse(url="/admin", status_code=303)
    else:
        raise HTTPException(status_code=404, detail='User not found')
