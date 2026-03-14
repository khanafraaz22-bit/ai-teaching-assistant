from fastapi import APIRouter
from app.api.v1.endpoints import auth, courses, chat, learning

api_router = APIRouter()

api_router.include_router(auth.router,     prefix="/auth",     tags=["Auth"])
api_router.include_router(courses.router,  prefix="/courses",  tags=["Courses"])
api_router.include_router(chat.router,     prefix="/chat",     tags=["Chat"])
api_router.include_router(learning.router, prefix="/learning", tags=["Learning Tools"])
