from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import google.generativeai as genai
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging
from datetime import datetime
import os
from flask import Flask


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Medical Image Analysis Bot", description="Powered by Medi AI Studio")

templates = Jinja2Templates(directory="templates")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("API_KEY is not set in the .env file. Get your free key from https://makersuite.google.com/app/apikey")

genai.configure(api_key=GOOGLE_API_KEY)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_and_query")
async def upload_and_query(image: UploadFile = File(...), query: str = Form(...)):
    try:

        image_content = await image.read()
        if not image_content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")


        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()

            img = Image.open(io.BytesIO(image_content))
            logger.info(f"Image Verifyed: {image.filename}")
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        medical_prompt = f"""

"""
        model = genai.GenerativeModel('gemini-1.5-flash')

        try:
            logger.info("Sending request to Medi Ai...")
            response = model.generate_content([medical_prompt, img])
            
            analysis = response.text
            
            if not analysis:
                raise Exception("Empty response from Medi AI")
            
            result = {
                "success": True,
                "model_info": {
                    "name": "MediClick",
                    "provider": "Google AI Studio",
                },
                "analysis": {
                    "content": analysis,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_filename": image.filename,
                    "query": query
                },
                "status": "completed"
            }
            
            logger.info("Medi Ai medical analysis completed successfully")
            return JSONResponse(status_code=200, content=result)

        except Exception as e:
            error_message = str(e)
            logger.error(f"Medi AI API ERROR: {error_message}")
            
            result = {
                "success": False,
                "model_info": {
                    "name": "MediClick",
                    "provider": "Google AI Studio",
                },
                "error": {
                    "message": error_message,
                    "suggestions": [
                    ],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "status": "error"
            }
            
            return JSONResponse(status_code=200, content=result)

    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f" error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Undefined error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API setup and service status"""
    try:

        model = genai.GenerativeModel('Medi AI')
        
        test_response = model.generate_content("Test connection - please respond with 'API Working'")
        
        return {
            "status": "healthy",
            "service": "Medical Image Analysis Bot",
            "model": "Gemini 1.5 Flash",
            "provider": "Google AI Studio",
            "api_configured": bool(GOOGLE_API_KEY),
            "api_test": "success",
            "test_response": test_response.text[:50] + "..." if test_response.text else "No response",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {
            "status": "error",
            "service": "Medical Image Analysis Bot",
            "error": str(e),
            "api_configured": bool(GOOGLE_API_KEY),
            "setup_url": "https://makersuite.google.com/app/apikey",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

@app.get("/info")
async def service_info():
    """Service information endpoint"""
    return {
        "service": "Medical Image Analysis Bot",
        "version": "2.0",
        "provider": "Google AI Studio",
        "model": "Gemini 1.5 Flash",
        "features": [
            "Medical image analysis",
            "Diagnostic suggestions", 
            "Severity assessment",
            "Professional recommendations",
            "Medical disclaimers"
        ],
        "supported_formats": ["JPG", "JPEG", "PNG", "GIF"],
        "max_file_size": "10MB",
        "response_time": "5-15 seconds",
    }



app = Flask(__name__)

@app.route('/')
def home():
    return "Hello from Flask on Render!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
