import multiprocessing

from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import models
from database import engine
from routers import auth, admin, camfeed, user
from starlette.staticfiles import StaticFiles
from routers.auth import get_current_user
import mp_shared

app = FastAPI()

models.Base.metadata.create_all(bind=engine)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(camfeed.router)
app.include_router(user.router)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, user=Depends(get_current_user)):
    return templates.TemplateResponse("home.html", {"request": request, "user": user})


if __name__ == '__main__':
    mp_shared.manager = multiprocessing.Manager()
    mp_shared.manager_detection_event = mp_shared.manager.Event()

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
