from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.routes import IndexerRouter
import uvicorn
from config import IndexerConfig
from fastapi import APIRouter
from services.indexer import Indexer
import threading
import pathway as pw

# Initialize FastAPI app
app = FastAPI(title="Indexer API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create API router
api_router = APIRouter()

# Initialize services and router
config = IndexerConfig()
indexer_router = IndexerRouter(
    api_router
)  # Passing None for now since IndexerService isn't implemented yet

# Include router
app.include_router(api_router, prefix="/api")


# Health check endpoint
@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "healthy"})


if __name__ == "__main__":
    # Create and start the server thread
    server_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": "main:app", "host": "0.0.0.0", "port": 8000, "reload": False},
    )
    server_thread.daemon = (
        True  # Make thread daemon so it exits when main program exits
    )
    server_thread.start()

    # Initialize indexer and run pathway
    indexer = Indexer(config)
    pw.run_all(terminate_on_error=False)
