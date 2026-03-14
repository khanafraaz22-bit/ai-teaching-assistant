import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.security import hash_password

async def fix():
    client = AsyncIOMotorClient('mongodb://localhost:27017')
    db = client['yt_course_assistant']
    users = db['users']
    new_hash = hash_password('testpassword123')
    await users.update_one(
        {'email': 'test@example2.com'},
        {'$set': {'hashed_password': new_hash}}
    )
    print('Password re-hashed OK')
    client.close()

asyncio.run(fix())