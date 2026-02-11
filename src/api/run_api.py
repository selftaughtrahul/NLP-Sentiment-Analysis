"""
Run the FastAPI application
"""
import uvicorn

if __name__ == "__main__":
    # Run this from the apps directory using: python -m src.api.run_api
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )