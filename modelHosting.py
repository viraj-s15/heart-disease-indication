from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import uvicorn
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio

app = FastAPI()


origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelInput(BaseModel):
    BMI: int
    Smoking: bool
    AlcoholDrinking: bool
    Stroke: bool
    PhysicalHealth: int
    MentalHealth: int
    DiffWalking: bool
    Sex: bool
    Race: int
    Diabetic: int
    PhysicalActivity: bool
    GenHealth: int
    SleepTime: int
    Asthma: bool
    KidneyDisease: bool
    SkinCancer: bool
    Age: int


filename = "./models/heartDiseaseModel.sav"
heart_disease_model = pickle.load(open(filename, "rb"))


@app.post("/heart-disease-prediction")
def prediciton(input_params: ModelInput):
    input_data = input_params.json()
    input_dictionary = json.loads(input_data)

    bmi = input_dictionary["BMI"]
    smoking = input_dictionary["Smoking"]
    alcohol_drinking = input_dictionary["Alcohol Drinking"]
    stroke = input_dictionary["Stroke"]
    physical_health = input_dictionary["Physical Health"]
    mental_health = input_dictionary["Mental Health"]
    diff_walking = input_dictionary["Difficult Walking"]
    sex = input_dictionary["Sex"]
    race = input_dictionary["Race"]
    diabetic = input_dictionary["Diabetic"]
    physical_activity = input_dictionary["Physical Activity"]
    general_health = input_dictionary["General Health"]
    sleep_time = input_dictionary["Sleep Time"]
    asthma = input_dictionary["Asthma"]
    kidney_disease = input_dictionary["Kidney Disease"]
    skin_cancer = input_dictionary["Skin Cancer"]
    age = input_dictionary["Age"]

    params_list = [
        bmi,
        smoking,
        alcohol_drinking,
        stroke,
        physical_health,
        mental_health,
        diff_walking,
        sex,
        race,
        diabetic,
        physical_activity,
        general_health,
        sleep_time,
        asthma,
        kidney_disease,
        skin_cancer,
        age,
    ]

    prediction = heart_disease_model.predict([params_list])

    if prediction[0] == 0:
        print(
            "The person with the given attribuites has a higher probability of not having any heart diseases"
        )
    else:
        print(
            "The person with the given attribuites has a higher probability of having any heart diseases"
        )


ngrok_tunnel = ngrok.connect(8000)
nest_asyncio.apply()
uvicorn.run(app, port=8000)
