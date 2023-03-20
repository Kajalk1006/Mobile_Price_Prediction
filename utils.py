import numpy as np
import pickle
import json

class MobilePricePrediction():

    def __init__(self,data):
        self.data = data
        print(self.data)

    def __loading(self): # private method

        with open('artifacts/scale.pkl','rb') as file:
            self.scaler = pickle.load(file)

        with open('artifacts/model.pkl','rb') as file:
            self.model = pickle.load(file)

        with open('artifacts/project_data.json','r') as file:
            self.project_data = json.load(file)

    def get_mobile_price_prediction(self):  # Public method 
     
        self.__loading()

        weight = self.data['Weight']
        resoloution = self.data['Resoloution']
        ppi = self.data['PPI']
        cpu_core = self.data['Cpu_core']
        cpu_freq = self.data['Cpu_freq']
        internal_mem = self.data['Internal_mem']
        ram = self.data['Ram']
        RearCam = self.data['rearCam']
        Front_Cam = self.data['front_Cam']
        battery = self.data['Battery']
        thickness = self.data['Thickness']

        user_data = np.zeros(len(self.project_data['column_names']))
        user_data[0] = weight
        user_data[1] = resoloution
        user_data[2] = ppi
        user_data[3] = cpu_core
        user_data[4] = cpu_freq
        user_data[5] = internal_mem
        user_data[6] = ram
        user_data[7] = RearCam
        user_data[8] = Front_Cam
        user_data[9] = battery
        user_data[10] = thickness


        user_data_scale = self.scaler.transform([user_data])   #scaling the user data 
        return self.model.predict(user_data_scale)[0]
    
