from sqlalchemy import Boolean, Column, Integer, String, Enum, ForeignKey, DateTime
from database import Base


class Users(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum("admin","operator"), default="operator")
    approved = Column(Boolean, default=None)


class RtspUrls(Base):
    __tablename__ = "rtsp_urls"
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False)


class ResetToken(Base):
    __tablename__ = "reset_tokens"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    expiration_date = Column(DateTime)
