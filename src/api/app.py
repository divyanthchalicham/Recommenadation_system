# src/api/app.py

import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.models.hybrid import HybridRecommender
from src.ingestion.stream_handler import StreamingService
from src.database.mongo_handler import init_db
from src.config import API_HOST, API_PORT


app = FastAPI(
    title="E-commerce Recommendation API",
    description="API for product recommendations",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


recommender = HybridRecommender()
stream_service = StreamingService()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
   
    init_db()
    
    
    stream_service.load_data()
    
  
    recommender.train()
    
    print("API startup complete. Services initialized.")


app.include_router(router)


app.state.recommender = recommender
app.state.stream_service = stream_service

def main():
  
    uvicorn.run(
        "src.api.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )

if __name__ == "__main__":
    main()