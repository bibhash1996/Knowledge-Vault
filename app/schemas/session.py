from pydantic import BaseModel, EmailStr

class SessionCreate(BaseModel):
    email: EmailStr

class SessionResponse(BaseModel):
    id: int
    email: EmailStr

    model_config = {
        "from_attributes": True,
    }