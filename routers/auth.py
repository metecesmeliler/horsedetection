from fastapi import Depends, HTTPException, status, APIRouter, Request, Response, Form, Query
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.responses import RedirectResponse
from passlib.context import CryptContext
from datetime import datetime, timedelta
from email_utils import send_email
from sqlalchemy.orm import Session
from database import SessionLocal
from jose import jwt, JWTError
from typing import Optional
import secrets
import models
import sys
sys.path.append("..")


router = APIRouter(
    prefix="/auth",
    tags=["auth"],
    responses={401: {"user": "Not authorized"}}
)


SECRET_KEY = "KlgH6AzYDeZeGwD288to79I3vTHT8wp7"
ALGORITHM = "HS256"

templates = Jinja2Templates(directory="templates")
bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_bearer = OAuth2PasswordBearer(tokenUrl="token")


class LoginForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.email: Optional[str] = None
        self.password: Optional[str] = None

    async def create_oauth_form(self):
        form = await self.request.form()
        self.email = form.get("email")
        self.password = form.get("password")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_password_hash(password):
    return bcrypt_context.hash(password)


def verify_password(plain_password, hashed_password):
    return bcrypt_context.verify(plain_password, hashed_password)


def authenticate_user(email: str, password: str, db):
    user = db.query(models.Users).filter(models.Users.email == email).first()

    if not user or not verify_password(password, user.hashed_password) or not user.approved:
        return False
    return user


def create_access_token(email: str, user_id: int, role: str,
                        expires_delta: Optional[timedelta] = None):

    encode = {"sub": email, "id": user_id, "role": role}
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    encode.update({"exp": expire})
    return jwt.encode(encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(request: Request):
    try:
        token = request.cookies.get("access_token")
        if token is None:
            return None
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        user_id: int = payload.get("id")
        role: str = payload.get("role")
        if email is None or user_id is None:
            return None
        return {"email": email, "id": user_id, "role": role}
    except JWTError:
        return None


def send_registration_email_to_admins(email: str, name: str, admin_emails):
    subject = "At Arabası Takip Sistemi: Yeni Kullanıcı"
    html_content = f"<p> {name} isimli kullanıcı {email} email adresiyle kayıt oldu.</p>"
    plain_content = f"{name} isimli kullanıcı {email} email adresiyle kayıt oldu."

    for admin_email in admin_emails:
        send_email(admin_email, html_content, plain_content, subject)


def get_reset_token(token: str, db: Session = Depends(get_db)):
    return db.query(models.ResetToken).filter_by(token=token).first()


def mark_inactive_users():
    with SessionLocal() as db:
        inactive_threshold = datetime.now() - timedelta(minutes=60)
        inactive_users = db.query(models.Users).filter(models.Users.last_login < inactive_threshold).all()
        for user in inactive_users:
            user.is_active = False
            db.commit()


@router.post("/token")
async def login_for_access_token(response: Response, form_data: OAuth2PasswordRequestForm = Depends(),
                                 db: Session = Depends(get_db)):
    user = authenticate_user(form_data.email, form_data.password, db)
    if not user:
        return False
    token_expires = timedelta(minutes=60)
    token = create_access_token(user.email, user.id, user.role, expires_delta=token_expires)
    user.last_login = datetime.now()
    user.is_active = True
    db.commit()
    response.set_cookie(key="access_token", value=token, httponly=True)
    return True


@router.get("/", response_class=HTMLResponse)
async def authentication_page(request: Request, current_user: dict = Depends(get_current_user)):
    if current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("login.html", {"request": request})


@router.post("/", response_class=HTMLResponse)
async def login(request: Request, db: Session = Depends(get_db)):
    try:
        form = LoginForm(request)
        await form.create_oauth_form()
        response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)

        validate_user_cookie = await login_for_access_token(response=response, form_data=form, db=db)
        if not validate_user_cookie:
            msg = "Incorrect Email or Password or User not Approved Yet"
            return templates.TemplateResponse("login.html", {"request": request, "msg": msg})
        return response
    except HTTPException:
        msg = "Unknown Error"
        return templates.TemplateResponse("login.html", {"request": request, "msg": msg})


@router.get("/logout")
async def logout(request: Request, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    user_id = current_user['id']
    if user_id:
        user = db.query(models.Users).get(user_id)
        if user:
            user.is_active = False
            db.commit()
    msg = "Logout Successful"
    response = templates.TemplateResponse("login.html", {"request": request, "msg": msg})
    response.delete_cookie(key="access_token")
    return response


@router.get("/register", response_class=HTMLResponse)
async def register(request: Request, current_user: dict = Depends(get_current_user)):
    if current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("register.html", {"request": request})


@router.post("/register", response_class=HTMLResponse)
async def register_user(request: Request, email: str = Form(...), name: str = Form(...),
                        password: str = Form(...), password2: str = Form(...),
                        db: Session = Depends(get_db)):

    validation = db.query(models.Users).filter_by(email=email).first()

    if password != password2 or validation is not None:
        msg = "Invalid registration request"
        return templates.TemplateResponse("register.html", {"request": request, "msg": msg})

    user_model = models.Users()
    user_model.email = email
    user_model.name = name

    hash_password = get_password_hash(password)
    user_model.hashed_password = hash_password
    user_model.role = "operator"
    user_model.approved = False

    db.add(user_model)
    db.commit()

    admin_emails = [admin.email for admin in db.query(models.Users).filter_by(role='admin').all()]
    send_registration_email_to_admins(email, name, admin_emails)

    msg = "User successfully created"
    return templates.TemplateResponse("login.html", {"request": request, "msg": msg})


@router.get("/change_password", response_class=HTMLResponse)
async def change_password_page(request: Request, current_user: dict = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    return templates.TemplateResponse("change_password.html", {"request": request, "user": current_user})


@router.post("/change_password", response_class=HTMLResponse)
async def change_password(request: Request, current_user: dict = Depends(get_current_user),
                          db: Session = Depends(get_db), new_password: str = Form(...), new_password2: str = Form(...)):

    if new_password != new_password2:
        msg = "Passwords do not match"
        return templates.TemplateResponse("change_password.html",
                                          {"request": request, "user": current_user, "msg": msg})

    user_id = current_user['id']
    new_hashed_password = get_password_hash(new_password)
    db.query(models.Users).filter(models.Users.id == user_id).update({"hashed_password": new_hashed_password})
    db.commit()

    return RedirectResponse(url="/user", status_code=303)


@router.get("/forgot_password", response_class=HTMLResponse)
async def forgot_password_page(request: Request, current_user: dict = Depends(get_current_user)):
    if current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("forgot_password.html", {"request": request})


@router.post("/forgot_password", response_class=HTMLResponse)
async def forgot_password(request: Request, current_user: dict = Depends(get_current_user),
                          db: Session = Depends(get_db), email: str = Form(...)):
    if current_user:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)

    user = db.query(models.Users).filter_by(email=email).first()
    if user is None:
        msg = "Email not found"
        return templates.TemplateResponse("forgot_password.html", {"request": request, "msg": msg})
    token = secrets.token_urlsafe(32)
    expiration = datetime.utcnow() + timedelta(minutes=5)

    reset_token = models.ResetToken(token=token, user_id=user.id, expiration_date=expiration)
    db.add(reset_token)
    db.commit()

    reset_link = f"http://127.0.0.1:8000/auth/reset_password/{reset_token.token}"
    subject = "At Arabası Takip Sistemi: Şifre Resetleme"
    html_content = f"<p>Şifrenizi resetlemek için linke tıklayın: <a href='{reset_link}'>Şifreyi Resetle</a></p>"
    plain_content = f"Şifrenizi resetlemek için linke tıklayın: {reset_link}"
    send_email(user.email, html_content, plain_content, subject)

    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)


@router.get("/reset_password/{token}", response_class=HTMLResponse)
async def reset_password_page(request: Request, token: str, db: Session = Depends(get_db)):
    reset_token = get_reset_token(token, db)
    if not reset_token or reset_token.expiration_date < datetime.utcnow():
        msg = "Şifre resetleme süresi geçti."
        return templates.TemplateResponse("login.html", {"request": request, "msg": msg})

    return templates.TemplateResponse("reset_password.html", {"request": request})


@router.post("/reset_password/{token}", response_class=HTMLResponse)
async def reset_password(request: Request, token: str, new_password: str = Form(...), db: Session = Depends(get_db)):
    reset_token = get_reset_token(token, db)
    if not reset_token or reset_token.expiration_date < datetime.utcnow():
        msg = "Şifre resetleme süresi geçti."
        return templates.TemplateResponse("login.html", {"request": request, "msg": msg})

    user = db.query(models.Users).get(reset_token.user_id)
    new_hashed_password = get_password_hash(new_password)
    user.hashed_password = new_hashed_password
    db.delete(reset_token)
    db.commit()

    msg = "Şifre başarıyla değiştirildi."
    return templates.TemplateResponse("login.html", {"request": request, "msg": msg})


@router.get("/change_email", response_class=HTMLResponse)
async def change_email_page(request: Request, current_user: dict = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    return templates.TemplateResponse("change_email.html", {"request": request, "user": current_user})


@router.post("/change_email", response_class=HTMLResponse)
async def change_email(request: Request, current_user: dict = Depends(get_current_user),
                       email: str = Form(...), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail='Authentication Failed')

    user = db.query(models.Users).filter_by(email=email).first()
    if user:
        msg = "Bu Email adresini kullanamazsınız."
        return templates.TemplateResponse("change_email.html", {"request": request, "msg": msg})

    token = secrets.token_urlsafe(32)
    expiration = datetime.utcnow() + timedelta(minutes=5)
    reset_token = models.ResetToken(token=token, user_id=current_user["id"], expiration_date=expiration)
    db.add(reset_token)
    db.commit()

    reset_link = f"http://127.0.0.1:8000/auth/change_email/{reset_token.token}?email={email}"
    subject = "At Arabası Takip Sistemi: Email Değiştirme"
    html_content = (f"<p>Email adresinizi değiştirmek için linke tıklayın: "
                    f"<a href='{reset_link}'>Şifreyi Resetle</a><br>"
                    f"Eğer böyle bir istekte bulunmadıysanız bu mesajı görmezden gelin.</p>")
    plain_content = (f"Şifrenizi resetlemek linke gidin: {reset_link}\n"
                     f"Eğer böyle bir istekte bulunmadıysanız bu mesajı görmezden gelin.")
    send_email(email, html_content, plain_content, subject)

    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)


@router.get("/change_email/{token}", response_class=HTMLResponse)
async def change_email_confirm(request: Request, token: str, email: str = Query(...),
                               current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail='Authentication Failed')

    reset_token = get_reset_token(token, db)
    if not reset_token or reset_token.expiration_date < datetime.utcnow():
        msg = "Email değiştirme süresi geçti."
        return templates.TemplateResponse("home.html", {"request": request, "msg": msg, "user": current_user})

    user = db.query(models.Users).filter_by(id=current_user["id"]).first()
    if user:
        user.email = email
        db.delete(reset_token)
        db.commit()

    return RedirectResponse(url="/user", status_code=status.HTTP_302_FOUND)
