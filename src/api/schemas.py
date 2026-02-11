"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class PredictionRequest(BaseModel):
    """Request schema for single prediction"""
    text: str = Field(..., min_length=1, max_length=5000, 
                     description="Text to analyze")
    model: str = Field(default="bert", 
                      description="Model to use (naive_bayes, logistic_regression, bert)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This product is amazing!",
                "model": "bert"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction"""
    texts: List[str] = Field(..., min_items=1, max_items=1000,
                            description="List of texts to analyze")
    model: str = Field(default="bert")
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Great product!",
                    "Terrible service",
                    "It's okay"
                ],
                "model": "bert"
            }
        }


class SentimentProbabilities(BaseModel):
    """Sentiment probabilities"""
    positive: float
    neutral: float
    negative: float


class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    text: str
    sentiment: str
    confidence: float
    probabilities: SentimentProbabilities
    model_used: str
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction"""
    total: int
    results: List[PredictionResponse]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: List[str]