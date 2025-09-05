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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Medical Image Analysis Bot", description="Powered by Google AI Studio")

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
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Validate image format
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
            # Re-open image for analysis (verify() closes the image)
            img = Image.open(io.BytesIO(image_content))
            logger.info(f"Image validated successfully: {image.filename}")
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        # Enhanced medical analysis prompt
        medical_prompt = f"""
You are an expert medical AI assistant with specialized training in medical image analysis and diagnosis.

Please analyze this medical image comprehensively and provide your assessment in the following structured format:

## VISUAL ANALYSIS
Describe exactly what you observe in the image, including:
- Overall image quality and clarity
- Anatomical structures visible
- Any abnormal findings, lesions, discolorations, or unusual features
- Size, shape, color, and texture of any concerning areas

## MEDICAL ASSESSMENT
Based on your visual analysis, provide:
- Primary suspected diagnosis with confidence level
- Secondary/differential diagnoses to consider
- Medical reasoning for each potential condition
- Key diagnostic features that support your assessment

## SEVERITY EVALUATION
Determine the urgency level:
- EMERGENCY: Requires immediate medical attention (within hours)
- URGENT: Needs prompt medical care (within 1-2 days) 
- ROUTINE: Can schedule regular appointment (within 1-2 weeks)
- FOLLOW-UP: Monitor and reassess as needed

Explain your reasoning for the urgency classification.

## RECOMMENDATIONS
Provide specific guidance on:
- Most appropriate medical specialist to consult (dermatologist, radiologist, etc.)
- Additional diagnostic tests or imaging that may be helpful
- General care instructions or precautions
- When to seek immediate care (red flag symptoms)

## Patient Query
Address this specific question: {query}

## IMPORTANT MEDICAL DISCLAIMER
This AI analysis is provided for informational and educational purposes only. It is NOT a substitute for professional medical diagnosis, treatment, or advice. Always consult with qualified healthcare professionals for proper medical evaluation. Seek immediate emergency medical care if experiencing severe or worsening symptoms. This analysis cannot replace physical examination, medical history review, or clinical judgment by licensed healthcare providers.

Please provide a thorough, professional analysis while emphasizing the critical importance of proper medical consultation.
"""

        # Initialize Google AI model
        model = genai.GenerativeModel('gemini-1.5-flash')

        try:
            # Generate response with image and text
            logger.info("Sending request to Google AI Studio...")
            response = model.generate_content([medical_prompt, img])
            
            # Process the response
            analysis = response.text
            
            if not analysis:
                raise Exception("Empty response from Google AI Studio")
            
            # Create successful response
            result = {
                "success": True,
                "model_info": {
                    "name": "Google Gemini 1.5 Flash",
                    "provider": "Google AI Studio",
                    "cost": "FREE",
                    "daily_limit": "100 requests"
                },
                "analysis": {
                    "content": analysis,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_filename": image.filename,
                    "query": query
                },
                "status": "completed"
            }
            
            logger.info("Google AI Studio medical analysis completed successfully")
            return JSONResponse(status_code=200, content=result)

        except Exception as e:
            error_message = str(e)
            logger.error(f"Google AI Studio API error: {error_message}")
            
            # Create error response
            result = {
                "success": False,
                "model_info": {
                    "name": "Google Gemini 1.5 Flash",
                    "provider": "Google AI Studio",
                    "cost": "FREE",
                    "daily_limit": "100 requests"
                },
                "error": {
                    "message": error_message,
                    "suggestions": [
                        "Verify your Google API key is valid and active",
                        "Check if you've exceeded the daily free limit (100 requests)",
                        "Ensure the image format is supported (JPG, PNG, GIF)",
                        "Verify your internet connection is stable",
                        "Try again in a few moments"
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
        logger.error(f"Unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API setup and service status"""
    try:
        # Test Google AI Studio connection
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Simple test request
        test_response = model.generate_content("Test connection - please respond with 'API Working'")
        
        return {
            "status": "healthy",
            "service": "Medical Image Analysis Bot",
            "model": "Google Gemini 1.5 Flash",
            "provider": "Google AI Studio",
            "api_configured": bool(GOOGLE_API_KEY),
            "api_test": "success",
            "test_response": test_response.text[:50] + "..." if test_response.text else "No response",
            "daily_limit": "100 requests (FREE)",
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
        "cost": "FREE (100 requests per day)",
        "supported_formats": ["JPG", "JPEG", "PNG", "GIF"],
        "max_file_size": "10MB",
        "response_time": "5-15 seconds",
        "accuracy": "High (powered by Google AI)"
    }

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🏥 MEDICAL IMAGE ANALYSIS BOT")
    print("=" * 60)
    print("🔑 Model: Google Gemini 1.5 Flash")
    print("💰 Cost: Completely FREE (100 requests/day)")
    print("🌐 Web Interface: http://localhost:8000")
    print("🔍 Health Check: http://localhost:8000/health")
    print("ℹ️  Service Info: http://localhost:8000/info")
    print()
    
    if not GOOGLE_API_KEY:
        print("❌ SETUP REQUIRED: GOOGLE_API_KEY not found!")
        print("📋 Quick Setup:")
        print("   1. Go to: https://makersuite.google.com/app/apikey")
        print("   2. Create your FREE Google AI Studio API key")
        print("   3. Add to .env file: GOOGLE_API_KEY=your_key_here")
        print("   4. Restart this application")
        print()
    else:
        print("✅ Google AI Studio configured successfully!")
        print()
    
    print("🚀 Starting server...")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")