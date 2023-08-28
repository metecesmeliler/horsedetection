from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from apscheduler.schedulers.background import BackgroundScheduler
from routers.auth import get_current_user, mark_inactive_users
from routers import auth, admin, camfeed, user
from starlette.staticfiles import StaticFiles
from database import engine
import multiprocessing
import models
import mp_shared

app = FastAPI()

models.Base.metadata.create_all(bind=engine)  # Database
templates = Jinja2Templates(directory="templates")  # Introducing templates directory
app.mount("/static", StaticFiles(directory="static"), name="static")  # Introducing static directory

# Including the routers
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(camfeed.router)
app.include_router(user.router)

# Creating a background scheduler for user activity check
scheduler = BackgroundScheduler()
scheduler.start()
scheduler.add_job(mark_inactive_users, trigger="interval", minutes=5)


# Main page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, user=Depends(get_current_user)):
    return templates.TemplateResponse("home.html", {"request": request, "user": user})


# Initializing shared multiprocessing declarations
if __name__ == '__main__':
    mp_shared.manager = multiprocessing.Manager()
    mp_shared.manager_detection_event = mp_shared.manager.Event()
    mp_shared.manager_queue = mp_shared.manager.Queue()

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
