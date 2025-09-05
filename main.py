from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import google.generativeai as genai
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Google AI Studio configuration (FREE)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the .env file. Get your free key from https://makersuite.google.com/app/apikey")

# Configure Google AI Studio
genai.configure(api_key=GOOGLE_API_KEY)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_and_query")
async def upload_and_query(image: UploadFile = File(...), query: str = Form(...)):
    try:
        # Read and validate image
        image_content = await image.read()
        if not image_content:
            raise HTTPException(status_code=400, detail="Empty file")

        # Validate image format
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
            # Re-open image for analysis (verify() closes the image)
            img = Image.open(io.BytesIO(image_content))
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        # Enhanced medical analysis prompt
        medical_prompt = f"""
        You are an expert medical AI assistant specialized in analyzing medical images and symptoms. 

        Please analyze this medical image and provide a comprehensive assessment:

        **1. VISUAL OBSERVATIONS:**
        - Describe what you can see in the image in detail
        - Note any visible abnormalities, lesions, or unusual features
        - Comment on image quality and clarity

        **2. POTENTIAL MEDICAL CONDITIONS:**
        - List possible diagnoses based on visual findings
        - Explain the reasoning for each potential condition
        - Indicate confidence levels for each assessment

        **3. SEVERITY ASSESSMENT:**
        - Assess if immediate medical attention might be needed
        - Categorize as: Emergency / Urgent / Routine consultation needed

        **4. RECOMMENDED ACTIONS:**
        - Suggest appropriate medical consultations (dermatologist, radiologist, etc.)
        - Recommend additional tests or examinations if needed
        - Provide general care recommendations

        **Patient's Question:** {query}

        🚨 **IMPORTANT MEDICAL DISCLAIMERS:**
        - This analysis is for informational and educational purposes only
        - This is NOT a substitute for professional medical diagnosis or treatment
        - Always consult qualified healthcare professionals for proper medical evaluation
        - Seek immediate emergency medical care if symptoms are severe or worsening
        - This AI analysis cannot replace physical examination and clinical judgment
        - Do not delay seeking professional medical care based on this analysis

        Please provide a thorough but clear analysis that emphasizes the importance of professional medical consultation.
        """

        # Initialize Google AI model
        model = genai.GenerativeModel('gemini-1.5-flash')

        try:
            # Generate response with image and text
            response = model.generate_content([medical_prompt, img])
            
            # Process the response
            analysis = response.text
            
            responses = {
                "google_ai_studio": {
                    "model": "Google Gemini 1.5 Flash (FREE)",
                    "status": "success",
                    "analysis": analysis,
                    "usage_info": "100 requests per day - completely FREE!",
                    "timestamp": str(io.BytesIO())
                }
            }
            
            logger.info("Google AI Studio medical analysis completed successfully")
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Google AI Studio API error: {error_message}")
            
            responses = {
                "google_ai_studio": {
                    "model": "Google Gemini 1.5 Flash (FREE)",
                    "status": "error",
                    "analysis": f"Error occurred: {error_message}\n\nPlease check:\n1. Your API key is valid\n2. You haven't exceeded the free daily limit (100 requests)\n3. The image format is supported\n4. Your internet connection is stable",
                    "usage_info": "100 requests per day - completely FREE!"
                }
            }

        return JSONResponse(status_code=200, content=responses)

    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API setup"""
    try:
        # Simple test to check if API is configured
        model = genai.GenerativeModel('gemini-1.5-flash')
        return {
            "status": "healthy",
            "service": "Google AI Studio Medical Bot",
            "model": "Gemini 1.5 Flash",
            "api_configured": bool(GOOGLE_API_KEY),
            "daily_limit": "100 requests (FREE)"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "setup_url": "https://makersuite.google.com/app/apikey"
        }

if __name__ == "__main__":
    import uvicorn
    print("🏥 Starting Medical Bot with Google AI Studio (FREE)...")
    print("🔑 Model: Google Gemini 1.5 Flash")
    print("💰 Cost: Completely FREE (100 requests/day)")
    print("🌐 Access your bot at: http://localhost:8000")
    print("🔍 Health check: http://localhost:8000/health")
    print()
    if not GOOGLE_API_KEY:
        print("❌ ERROR: GOOGLE_API_KEY not found!")
        print("📋 Setup instructions:")
        print("   1. Go to: https://makersuite.google.com/app/apikey")
        print("   2. Create your FREE API key")
        print("   3. Add to .env file: GOOGLE_API_KEY=your_key_here")
        print()
    else:
        print("✅ Google AI Studio configured successfully!")
        print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)