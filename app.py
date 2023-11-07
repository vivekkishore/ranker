from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('new_model.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


Minmaxnormalizera = MinMaxScaler()
df_act=pd.read_csv('pjrank1.csv')
df_encoded=pd.read_csv('pjrank1_encoded.csv')


@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':

        # Encoding of variables
        meeting_details= request.form['meeting_detail']
        encoder=LabelEncoder()
        encoder.fit(df_act['meeting details'])
        meeting_details_ready=encoder.transform([meeting_details])[0]

        Job_profile=request.form['job_profile']
        encoder1=LabelEncoder()
        encoder1.fit(df_act['Job profile'])
        Job_profile_ready=encoder1.transform([Job_profile])[0]

        Job_family=request.form['job_family']
        encoder2=LabelEncoder()
        encoder2.fit(df_act['Job family'])
        Job_family_ready=encoder2.transform([Job_family])[0]

        Overall_experience_ready=int(request.form['overall_exp'])
        if (Overall_experience_ready>29):
            Overall_experience_ready=1
        else:
            Minmaxnormalizer2=MinMaxScaler()
            Minmaxnormalizer2.fit(pd.DataFrame(df_encoded['Overall experience(in years)']))
            Overall_experience_ready=Minmaxnormalizer2.transform([[Overall_experience_ready]])[0][0]

        Primary_Skill=request.form['primary_skill']
        encoder3=LabelEncoder()
        encoder3.fit(df_act['Primary Skill'])
        Primary_Skill_ready=encoder3.transform([Primary_Skill])[0]

        secondary_Skill=request.form['secondary_skill']
        encoder4=LabelEncoder()
        encoder4.fit(df_act['secondary Skill'])
        secondary_Skill_ready=encoder4.transform([secondary_Skill])[0]

        certifications=request.form['certification']
        encoder5=LabelEncoder()
        encoder5.fit(df_act['certifications'])
        certifications_ready=encoder5.transform([certifications])[0]

        Org_Experience_ready=int(request.form['org_experience'])

        if (Org_Experience_ready>19):
            Org_Experience_ready=1
        else:
            Minmaxnormalizer1=MinMaxScaler()
            Minmaxnormalizer1.fit(pd.DataFrame(df_encoded['Org Experience']))
            Org_Experience_ready=Minmaxnormalizer1.transform([[Org_Experience_ready]])[0][0]



        TQ_score_ready=request.form['score']
        
        

        Employment_type=request.form['emp_type']
        Employment_type_ready=0

        if (Employment_type=='full time'):
            Employment_type_ready=1
        else:
            Employment_type_ready=0


        Management_level= request.form['level']
        
        level_dict={'12-Associate':1,'11-Analyst':2,'10-Senior Analyst':3,'9-Team Lead/Consultant':4,'8-Associate Manager':5,'7-Manager':6,'6-Senior Manager':7,'5 and above':7}
        for key,value in level_dict.items():
             if Management_level==key:
               Management_level=value
               break
             
        Management_level_ready=Management_level

        Specialization=request.form['specialization']
        encoder7=LabelEncoder()
        encoder7.fit(df_act['Specialization'])
        Specialization_ready=encoder7.transform([Specialization])[0]

        Trainings=request.form['training']
        encoder8=LabelEncoder()
        encoder8.fit(df_act['Trainings'])
        Trainings_ready=encoder8.transform([Trainings])[0]

        Highest_qualification=request.form['qualification']
        encoder9=LabelEncoder()
        encoder9.fit(df_act['Highest qualification'])
        Highest_qualification_ready=encoder9.transform([Highest_qualification])[0]

        countrycode=request.form['country']
        encoder10=LabelEncoder()
        encoder10.fit(df_act['countrycode'])
        countrycode_ready=encoder10.transform([countrycode])[0]

        Meeting_required_optional=request.form['status']
        Meeting_required_optional_ready=0

        if (Meeting_required_optional=='required'):
            Meeting_required_optional_ready=1
        else:
            Meeting_required_optional_ready=0

        Email_mention=request.form['mention']
        Email_mention_ready=0

        if (Email_mention=='yes'):
            Email_mention_ready=1
        else:
            Email_mention_ready=0

        meeting_type=request.form['series']
        meeting_type_ready=0

        if (meeting_type=='instance'):
            meeting_type_ready=1
        else:
            meeting_type_ready=0


        recent_emails =request.form['similar']
        recent_emails_ready=0

        if (recent_emails=='yes'):
            recent_emails_ready=1
        else:
            recent_emails_ready=0

        # scaling of variables

        columns_to_scale = ['meeting details','Job profile','Job family','Primary Skill','secondary Skill','certifications','TQ score','Management level','Specialization','Trainings','Highest qualification','countrycode']
        values_to_scale1=[meeting_details_ready,Job_profile_ready,Job_family_ready,Primary_Skill_ready,secondary_Skill_ready,certifications_ready,TQ_score_ready, Management_level_ready,Specialization_ready,Trainings_ready,Highest_qualification_ready,countrycode_ready]

        # Use a loop to scale each column
        for i in range(1,13):
          Minmaxnormalizera.fit(pd.DataFrame(df_encoded[columns_to_scale[i-1]]))
          values_to_scale1[i-1]=Minmaxnormalizera.transform([[values_to_scale1[i-1]]])[0][0]

        meeting_details_ready = values_to_scale1[0]
        Job_profile_ready = values_to_scale1[1]
        Job_family_ready = values_to_scale1[2]
        Primary_Skill_ready = values_to_scale1[3]
        secondary_Skill_ready = values_to_scale1[4]
        certifications_ready = values_to_scale1[5]
        TQ_score_ready = values_to_scale1[6]
        Management_level_ready = values_to_scale1[7]
        Specialization_ready = values_to_scale1[8]
        Trainings_ready = values_to_scale1[9]
        Highest_qualification_ready = values_to_scale1[10]
        countrycode_ready = values_to_scale1[11]
        print(TQ_score_ready)

        prediction=model.predict([[meeting_details_ready,Job_profile_ready,Job_family_ready,Overall_experience_ready,Primary_Skill_ready,secondary_Skill_ready,certifications_ready,Org_Experience_ready,TQ_score_ready,Employment_type_ready,Management_level_ready,Specialization_ready,Trainings_ready,
                                   Highest_qualification_ready,countrycode_ready,Meeting_required_optional_ready,Email_mention_ready,meeting_type_ready, recent_emails_ready]])
        
        
        output=round(prediction[0],3)
        if output<0:
            return render_template('index.html',prediction_text="The calculated rank score is 0")
        else:
            return render_template('index.html',prediction_text="The calculated rank score is {}".format(output))


    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

