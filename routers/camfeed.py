from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from starlette.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from routers.auth import get_current_user
from sqlalchemy.orm import Session
from models import RtspUrls, Users
from email_utils import send_email
from database import SessionLocal
from typing import Annotated
import mediapipe as mp
from mediapipe.tasks import python
import multiprocessing
import numpy as np
import mp_shared
import datetime
import cv2
import time

router = APIRouter(
    prefix='/camfeed',
    tags=['camfeed']
)

templates = Jinja2Templates(directory="templates")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Global variables, and lists; user, and database dependencies
db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]
prev_rtsp_urls = []
roi_settings_list = []
email_list = []
num_cameras = 0
detection_process = None


# Class declaration for Region of Interest Settings
class RoiSettings:
    def __init__(self, x1=0, y1=0, x2=0, y2=0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


# Function for sending email
def send_detection_email(cam_id):
    global email_list
    subject = "At Arabası Takip Sistemi: At Arabası Algılandı"
    html_content = f"<p> {cam_id} numaralı kamerada at arabası algılandı.</p>"
    plain_content = f"{cam_id} numaralı kamerada at arabası algılandı."
    for approved_email in email_list:
        try:
            send_email(approved_email, html_content, plain_content, subject)
        except Exception as e:
            print(f"An exception occurred in send_email: {e}")


# Generator function that generates frames
def generate_frames(cap, roi_settings):
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not (roi_settings.x1 == roi_settings.x2 == roi_settings.y1 == roi_settings.y2) or roi_settings.x1 != 0:
                frame = frame[roi_settings.y1:roi_settings.y2, roi_settings.x1:roi_settings.x2]
            _, img_encoded = cv2.imencode(".jpg", frame)
            img_byte = img_encoded.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + img_byte + b"\r\n")
    except Exception as e:
        print(f"An exception occurred in stream_video: {e}")


# Object detection function that creates the Mediapipe model
def detect_objects(url, process_id, roi_settings, detection_logs, detection_event):
    print("Inside detect_objects")
    last_timestamp_ms = None
    detection_time_last = datetime.datetime.min
    try:
        # faster
        # model_path = r'C:\Documents\efficientdet_lite0.tflite'
        # more accuracy
        model_path = r'C:\Documents\efficientdet_lite2.tflite'
        BaseOptions = mp.tasks.BaseOptions
        DetectionResult = mp.tasks.components.containers.DetectionResult
        ObjectDetector = mp.tasks.vision.ObjectDetector
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        def print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
            nonlocal detection_time_last
            nonlocal detection_logs
            for i, detection in enumerate(result.detections):
                print(f'Process: {process_id} Detection #{i + 1}:')

                bounding_box = detection.bounding_box
                box_width = bounding_box.width
                box_height = bounding_box.height
                box_origin_x = bounding_box.origin_x
                box_origin_y = bounding_box.origin_y
                print(f'  Box: (x: {box_origin_x}, y: {box_origin_y}, w: {box_width}, h: {box_height})')

                category = detection.categories[0]
                category_score = category.score
                category_name = category.category_name
                if category_name == 'horse':
                    print("###################### Horse detected ######################")
                elif category_name == 'person':
                    print("********************** Person Detected **********************")
                    try:
                        current_time = datetime.datetime.now()
                        if (current_time - detection_time_last).total_seconds() >= 15:
                            detection_time_last = current_time
                            detection_logs.put("İnsan algılandı! Kamera " + str(process_id) + ": "
                                               + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    except Exception as exc:
                        print(f"An exception occurred in print_result {process_id}: {exc}")
                print(f'  Category Score: {category_score}')
                print(f'  Category Name: {category_name}')
                print()

        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            score_threshold=0.1,
            max_results=5,
            result_callback=print_result)

        with ObjectDetector.create_from_options(options) as detector:
            try:
                camera_id = int(url)
                video_capture = cv2.VideoCapture(camera_id)
            except ValueError:
                video_capture = cv2.VideoCapture(url)
            while detection_event.is_set():
                ret, frame = video_capture.read()
                if not ret:
                    print("Error when getting the frame")
                    break

                if not (roi_settings.x1 == roi_settings.x2 == roi_settings.y1 == roi_settings.y2)\
                        or roi_settings.x1 != 0:
                    frame = frame[roi_settings.y1:roi_settings.y2, roi_settings.x1:roi_settings.x2]
                frame_np = np.array(frame)
                frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                frame_timestamp_ms = int(1000 * time.time())
                if last_timestamp_ms is not None and frame_timestamp_ms <= last_timestamp_ms:
                    frame_timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = frame_timestamp_ms
                detector.detect_async(mp_image, frame_timestamp_ms)

    except Exception as e:
        print(f"An exception occurred in process {process_id}: {e}")


# Function that creates a multiprocessing pool for parallel object detection
def concurrent_detection(prev_rtsp_url_list, roi_setting_list, detection_logs, detection_event):
    try:
        with multiprocessing.Pool(processes=len(prev_rtsp_url_list)) as pool:
            async_results = []
            for url, process_id in zip(prev_rtsp_url_list, range(len(prev_rtsp_url_list))):
                roi_settings = roi_setting_list[process_id]
                async_result = pool.apply_async(detect_objects,
                                                args=(url, process_id + 1, roi_settings,
                                                      detection_logs, detection_event))
                async_results.append(async_result)
            for async_result in async_results:
                async_result.get()
    except Exception as e:
        print(f"An exception occurred in concurrent_detection: {e}")
    finally:
        async_results.clear()


# Generator function that generates detection log message
def generate_detection_log():
    if mp_shared.manager_queue:
        latest_message = mp_shared.manager_queue.get()
        yield f"data: {latest_message}\n\n"
    else:
        return None


# '/' endpoint that fetches urls from database and renders camfeed.html template
@router.get("/", response_class=HTMLResponse)
async def camfeed(user: user_dependency, db: db_dependency, request: Request):
    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    global prev_rtsp_urls
    global roi_settings_list
    global email_list

    email_list.clear()
    rtsp_urls = db.query(RtspUrls.url).all()
    rtsp_urls = [url[0] for url in rtsp_urls]
    email_list = [approved.email for approved in db.query(Users).filter_by(approved=True).all()]
    if rtsp_urls != prev_rtsp_urls:
        for _ in rtsp_urls:
            roi_settings_list.append(RoiSettings())
        prev_rtsp_urls = rtsp_urls

    global num_cameras
    num_cameras = len(prev_rtsp_urls)
    return templates.TemplateResponse("camfeed.html",
                                      {"request": request, "num_cameras": num_cameras, "user": user})


# '/stream_camera/{cam_index}' endpoint that gets video captures from urls and returns streaming response
@router.get("/stream_camera/{cam_index}")
async def stream_camera(user: user_dependency, cam_index: int):
    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    global prev_rtsp_urls
    if cam_index < 1 or cam_index > len(prev_rtsp_urls):
        raise HTTPException(status_code=400, detail="Invalid camera index")

    roi_settings = roi_settings_list[cam_index - 1]
    url = prev_rtsp_urls[cam_index - 1]
    try:
        camera_id = int(url)
        cap = cv2.VideoCapture(camera_id)
    except ValueError:
        cap = cv2.VideoCapture(url)
    return StreamingResponse(generate_frames(cap, roi_settings), media_type="multipart/x-mixed-replace;boundary=frame")


# '/adjust_roi/{cam_index}' (get) endpoint that gets the adjust roi page
@router.get("/adjust_roi/{cam_index}", response_class=HTMLResponse)
async def get_adjust_roi_page(request: Request, user: user_dependency, cam_index: int):
    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    global prev_rtsp_urls
    if cam_index < 1 or cam_index > len(prev_rtsp_urls):
        raise HTTPException(status_code=400, detail="Invalid camera index")
    url = prev_rtsp_urls[cam_index - 1]
    try:
        camera_id = int(url)
        cap = cv2.VideoCapture(camera_id)
    except ValueError:
        cap = cv2.VideoCapture(url)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return templates.TemplateResponse("adjust_roi.html",
                                      {"user": user, "request": request, "cam_index": cam_index,
                                       "frame_width": frame_width, "frame_height": frame_height})


# '/adjust_roi/{cam_index}' (post) endpoint that gets the form information and creates a roi_settings object
@router.post("/adjust_roi/{cam_index}")
async def adjust_roi(request: Request, user: user_dependency, cam_index: int):
    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    global prev_rtsp_urls
    if cam_index < 1 or cam_index > len(prev_rtsp_urls):
        raise HTTPException(status_code=400, detail="Invalid camera index")

    form_data = await request.form()
    x1 = int(form_data.get("x1"))
    y1 = int(form_data.get("y1"))
    x2 = int(form_data.get("x2"))
    y2 = int(form_data.get("y2"))

    roi_settings = roi_settings_list[cam_index - 1]
    url = prev_rtsp_urls[cam_index - 1]
    try:
        camera_id = int(url)
        cap = cv2.VideoCapture(camera_id)
    except ValueError:
        cap = cv2.VideoCapture(url)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    roi_settings.x1 = max(0, min(x1, int(frame_width) - 1))
    roi_settings.y1 = max(0, min(y1, int(frame_height) - 1))
    roi_settings.x2 = max(0, min(x2, int(frame_width) - 1))
    roi_settings.y2 = max(0, min(y2, int(frame_height) - 1))

    return RedirectResponse(url="/camfeed", status_code=303)


# '/start_detection' endpoint that starts object detection by calling concurrent_detection function (multiprocessing)
@router.post("/start_detection", response_class=JSONResponse)
async def start_detection(request: Request, user: user_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    global detection_process
    global prev_rtsp_urls
    global roi_settings_list
    msg = "Takip zaten çalışıyor"
    if detection_process is None or not detection_process.is_alive():
        mp_shared.manager_detection_event.set()
        detection_process = multiprocessing.Process(target=concurrent_detection,
                                                    args=(prev_rtsp_urls, roi_settings_list, mp_shared.manager_queue,
                                                          mp_shared.manager_detection_event,))
        detection_process.start()
        msg = None
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return {"timestamp": timestamp, "warning": msg}


# '/stop_detection' endpoint that ends object detection
@router.post("/stop_detection", response_class=JSONResponse)
async def stop_detection(request: Request, user: user_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    global detection_process
    msg = "Takip zaten sona erdi!"
    if detection_process is not None and detection_process.is_alive():
        mp_shared.manager_detection_event.clear()
        detection_process.join()
        detection_process = None
        msg = None
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return {"timestamp": timestamp, "warning": msg}


# '/get_detection_log' endpoint that returns streaming response for the detection log (Server-Sent Events)
@router.get("/get_detection_log")
async def get_detection_log_sse():
    return StreamingResponse(generate_detection_log(), media_type="text/event-stream")
