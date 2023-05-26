from flask_sqlalchemy import SQLAlchemy

db=SQLAlchemy()

class UserModel(db.Model):
    __tablename__='Users'
    id= db.Column(db.Integer,primary_key=True,autoincrement=True)
    name=db.Column(db.String(80))
    email=db.Column(db.String(80))
    password=db.Column(db.String(150))
    phone=db.Column(db.String(20))
    created_at=db.Column(db.DateTime, default=db.func.now())
    def __init__(self,name,email,password,phone):
        self.name=name
        self.email=email
        self.password=password
        self.phone=phone
    def json(self):
        return {"name":self.name,"email":self.email,"password":self.password,"phone":self.phone}