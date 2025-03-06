"""
API routes for the recommendation system.
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

router = APIRouter()


class ProductRecommendation(BaseModel):
    product_id: str
    product_name: str
    score: float
    category: str
    price: float
    reason: str

class RecommendationResponse(BaseModel):
    user_id: str
    recommended_products: List[ProductRecommendation]

class UserActivity(BaseModel):
    user_id: str
    action_type: str
    product_id: str
    timestamp: Optional[str] = None
    category: str
    price: float

@router.get("/")
async def root():
    """Root endpoint to check API status."""
    return {"message": "E-commerce Recommendation API is running"}

@router.get("/users")
async def get_users():
    """Get a list of all users in the system."""
    from src.database.mongo_handler import get_all_users
    users = get_all_users()
    return {"count": len(users), "users": users[:50]}  # Limit to first 50 users

@router.get("/recommendation-status")
async def get_recommendation_status():
    """Get status of recommendations in the system."""
    from src.database.mongo_handler import db, COLLECTION_RECOMMENDATIONS
    users_with_recs = list(db[COLLECTION_RECOMMENDATIONS].distinct("user_id"))
    return {
        "users_with_recommendations": len(users_with_recs),
        "sample_users": users_with_recs[:5] if users_with_recs else []
    }

@router.post("/generate-recommendations/{user_id}")
async def generate_recommendations(user_id: str, request: Request):
    """Manually generate recommendations for a user."""
    recommender = request.app.state.recommender
    
    if not recommender:
        raise HTTPException(status_code=500, detail="Recommendation service not initialized")
    
    
    from src.database.mongo_handler import get_user_activity
    user_activities = get_user_activity(user_id=user_id)
    
    if not user_activities:
        raise HTTPException(status_code=404, detail=f"User {user_id} has no activity data")
    

    recommendations = recommender.generate_recommendations(user_id)
    
    return {
        "user_id": user_id,
        "activities_count": len(user_activities),
        "recommendations_generated": len(recommendations) if recommendations else 0,
        "message": "Recommendations generated successfully" if recommendations else "Failed to generate recommendations"
    }

@router.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(user_id: str, request: Request, limit: int = 10):
    """
    Get recommendations for a user.
    
    Args:
        user_id: User ID to get recommendations for
        limit: Maximum number of recommendations
    """
    recommender = request.app.state.recommender
    
    if not recommender:
        raise HTTPException(status_code=500, detail="Recommendation service not initialized")
    
    recommendations = recommender.get_formatted_recommendations(user_id, top_k=limit)
    
    if not recommendations["recommended_products"]:
        raise HTTPException(status_code=404, detail=f"No recommendations found for user {user_id}")
    
    return recommendations

@router.post("/activity")
async def add_activity(activity: UserActivity, request: Request):
    """
    Add a new user activity.
    
    Args:
        activity: User activity data
    """
    from src.database.mongo_handler import insert_activity
    

    if not activity.timestamp:
        activity.timestamp = datetime.now().isoformat()
    
  
    activity_dict = activity.dict()
    
   
    success = insert_activity(activity_dict)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to insert activity")
    
    return {"message": "Activity added successfully"}

@router.post("/start-streaming")
async def start_streaming(request: Request, speed_factor: int = 100):
    """
    Start streaming simulated user activity data.
    
    Args:
        speed_factor: How many times faster than real-time
    """
    stream_service = request.app.state.stream_service
    
    if not stream_service:
        raise HTTPException(status_code=500, detail="Streaming service not initialized")
    
    success = stream_service.start_streaming(speed_factor=speed_factor)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to start streaming")
    
    return {"message": f"Streaming started with speed factor {speed_factor}x"}

@router.post("/stop-streaming")
async def stop_streaming(request: Request):
    """Stop streaming simulated user activity data."""
    stream_service = request.app.state.stream_service
    
    if not stream_service:
        raise HTTPException(status_code=500, detail="Streaming service not initialized")
    
    success = stream_service.stop_streaming()
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to stop streaming")
    
    return {"message": "Streaming stopped"}