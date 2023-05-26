from flask import Flask,redirect,url_for,request,flash,jsonify
# from flask_sqlalchemy import SQLAlchemy
from flask_restful import Api,Resource,reqparse
# from models import Predict,db
# from sqlalchemy import select
from flask_cors import CORS
import pickle
import numpy as np
app=Flask(__name__)
CORS(app)
app.secret_key = 'Helloworld'
# app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///data.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False

api=Api(app)
# db.init_app(app)


# @app.before_first_request
# def create_table():
#     db.create_all()

model = pickle.load(open('model.pkl', 'rb'))

class predictModel(Resource):
 #def get(self):
  #   predict=Predict.query.all()
   #  return {'Predict':list(x.json() for x in predict)}
 def post(self):
    # Handle actual POST requests
    form_data = request.get_json()
#     new_predict = Predict(Gender=form_data['Gender'],marital_status=form_data['marital_status'],age=form_data['age'],scolarship=form_data["scolarship"],mother_qualification=form_data["mother_qualification"],mother_occupation=form_data['mother_occupation'],
#     father_occupation=form_data["father_occupation"],inflation_rate=form_data["inflation_rate"],previous_qualifcation=form_data["previous_qualifcation"],course=form_data["course"],
#     curricular_units_1st_sem_without_evaluations=form_data["curricular_units_1st_sem_without_evaluations"],curricular_units_1st_sem_approved=form_data["curricular_units_1st_sem_approved"],
#     curricular_units_1st_sem_credited=form_data["curricular_units_1st_sem_credited"],curricular_units_1st_sem_with_evaluations=form_data["curricular_units_1st_sem_with_evaluations"],
#    curricular_units_2nd_sem_approved=form_data["curricular_units_2nd_sem_approved"],
#     curricular_units_2nd_sem_credited=form_data["curricular_units_2nd_sem_credited"],curricular_units_2nd_sem_with_evaluations=form_data["curricular_units_2nd_sem_with_evaluations"],  curricular_units_2nd_sem_without_evaluations=form_data["curricular_units_2nd_sem_without_evaluations"])
#     db.session.add(new_predict)
#     db.session.commit()
#     db.session.flush()
    print(form_data.values())
    input_data = np.asarray(list(form_data.values())).reshape(1, -1)
    prediction = model.predict(input_data)
    return jsonify({'prediction': str(prediction[0])})

api.add_resource(predictModel,'/predict')   
app.debug=True  
if __name__ == '__main__':
    app.run(host='localhost',port=5000)