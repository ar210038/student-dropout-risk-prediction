"""Semester-1 student dropout early-warning Streamlit application."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
MODEL_PATH=ROOT/'models'/'uci_sem1'/'dropout_early_warning_model.joblib'
METADATA_PATH=ROOT/'models'/'uci_sem1'/'model_metadata.json'
THRESHOLD=0.48
SOURCE_SEMESTER_GRADE_MAX=18.875
FORM_FIELDS=['previous_academic_gpa_normalized','Age at enrollment',"Mother's occupation","Father's occupation",'Scholarship holder','Debtor','Tuition fees up to date','Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)','Curricular units 1st sem (approved)','Curricular units 1st sem (grade)']
OCCUPATIONS={
0:'Student',1:'Legislative/executive representatives, directors and managers',2:'Intellectual and scientific specialists',3:'Intermediate-level technicians and professions',4:'Administrative staff',5:'Personal services, security and sales workers',6:'Skilled agriculture, fisheries and forestry workers',7:'Skilled industry, construction and craft workers',8:'Machine operators and assembly workers',9:'Unskilled workers',10:'Armed Forces',90:'Other situation',99:'Unspecified',101:'Armed Forces officers',102:'Armed Forces sergeants',103:'Other Armed Forces personnel',112:'Administrative and commercial services directors',114:'Hotel, catering, trade and other service directors',121:'Physical sciences, mathematics and engineering specialists',122:'Health professionals',123:'Teachers',124:'Finance, administration and commercial specialists',125:'Information and communication technology specialists',131:'Intermediate science and engineering technicians',132:'Intermediate health technicians',134:'Intermediate legal, social, cultural and sports services',135:'Information and communication technology technicians',141:'Office, secretarial and data-processing workers',143:'Data, accounting, statistics and financial operators',144:'Other administrative support staff',151:'Personal service workers',152:'Sellers',153:'Personal care workers',154:'Protection and security personnel',161:'Market-oriented farmers and skilled agricultural workers',163:'Subsistence farmers, fishers, hunters and gatherers',171:'Skilled construction workers (except electricians)',172:'Skilled metallurgy and metalworking workers',173:'Printing, precision, jewellery and craft workers',174:'Skilled electrical and electronics workers',175:'Food, wood, clothing and other craft workers',181:'Fixed-plant and machine operators',182:'Assembly workers',183:'Vehicle drivers and mobile-equipment operators',191:'Cleaning workers',192:'Unskilled agriculture, fisheries and forestry workers',193:'Unskilled extractive, construction, manufacturing and transport workers',194:'Meal preparation assistants',195:'Street vendors and service providers'}
OCCUPATION_BY_LABEL={label:code for code,label in OCCUPATIONS.items()}

@st.cache_resource
def load_model_artifact()->dict[str,Any]: return joblib.load(MODEL_PATH)

def make_input_frame(values:dict[str,Any])->pd.DataFrame:
    if set(values)!=set(FORM_FIELDS): raise ValueError('Answers do not match the required 11-field form.')
    return pd.DataFrame([{field:values[field] for field in FORM_FIELDS}])

def validate_inputs(values:dict[str,Any])->list[str]:
    errors=[]
    if values['Curricular units 1st sem (approved)']>values['Curricular units 1st sem (enrolled)']: errors.append('Courses Passed cannot be greater than Courses Enrolled.')
    return errors

def dropout_probability(artifact:dict[str,Any],values:dict[str,Any])->float:
    pipeline=artifact['pipeline']; classes=list(pipeline.classes_); index=classes.index(1)
    return float(pipeline.predict_proba(make_input_frame(values))[0,index])

def risk_label(probability:float)->str: return 'Elevated Estimated Risk' if probability>=THRESHOLD else 'Lower Estimated Risk'

def semester_gpa_to_source_grade(semester_gpa:float)->float:
    """Map the visible 0–4 GPA to the frozen model's original grade range."""
    return float(semester_gpa)/4.0*SOURCE_SEMESTER_GRADE_MAX

def support_guidance(*,semester_gpa:float,enrolled:int,evaluations:int,passed:int,
                     debtor:bool,tuition_up_to_date:bool,scholarship:bool,
                     previous_gpa:float)->tuple[list[str],list[str]]:
    """Return prioritized, non-causal indicators and institutional support options."""
    pass_rate=passed/enrolled if enrolled else 0.0
    evaluation_rate=evaluations/enrolled if enrolled else 0.0
    candidates=[]
    if semester_gpa<2.0:
        candidates.append((10,f'Low recent academic performance — Semester GPA: {semester_gpa:.2f} / 4.00',['Academic advising','Tutoring / study support']))
    if not enrolled:
        candidates.append((9,'No course enrollment was reported',['Study-plan review']))
    elif pass_rate<0.5:
        candidates.append((9,f'Low course completion — {passed} of {enrolled} courses passed',['Course-specific academic support','Study-plan review']))
    elif pass_rate<0.7:
        candidates.append((5,f'Moderate course completion — {passed} of {enrolled} courses passed',['Study-plan review']))
    if not tuition_up_to_date:
        candidates.append((8,'Tuition fees are not currently up to date',['Financial-aid review','Installment/payment support if available']))
    if debtor:
        candidates.append((8,'Outstanding debtor status was reported',['Financial counselling','Review possible payment arrangements']))
    if enrolled and evaluation_rate<0.75:
        candidates.append((7,f'Low evaluation completion — {evaluations} evaluations for {enrolled} enrolled courses',['Student engagement follow-up','Check attendance / participation difficulties']))
    if not scholarship and (debtor or not tuition_up_to_date):
        candidates.append((4,'No scholarship was reported alongside a financial warning',['Review scholarship or student-aid eligibility']))
    if previous_gpa<3.0:
        candidates.append((3,f'Weaker previous academic background — HSC / Equivalent GPA: {previous_gpa:.2f} / 5.00',['Foundation/remedial academic support if needed']))
    selected=sorted(candidates,key=lambda item:item[0],reverse=True)[:3]
    supports=[]
    for _,_,recommendations in selected:
        if recommendations[0] not in supports: supports.append(recommendations[0])
    for _,_,recommendations in selected:
        for recommendation in recommendations[1:]:
            if recommendation not in supports and len(supports)<4: supports.append(recommendation)
    return [indicator for _,indicator,_ in selected],supports
def cards(items:list[tuple[str,str]])->None:
    for column,(label,value) in zip(st.columns(len(items)),items): column.metric(label,value)

def home()->None:
    st.title('Student Dropout Early Warning System')
    st.subheader('Machine Learning-Based Early Identification for Student Support')
    st.write("This academic prototype estimates a student's dropout risk using background, financial, and recent academic information. The aim is to support earlier academic or financial follow-up.")
    st.markdown('<div class="workflow"><span>Completed Semester</span><b>→</b><span>Student Information</span><b>→</b><span>Machine Learning</span><b>→</b><span>Dropout Risk Estimate</span><b>→</b><span>Early Support</span></div>',unsafe_allow_html=True)
    cards([('Model','Random Forest'),('Inputs','11'),('ROC-AUC','93.8%')])
    st.markdown('### Why Early Warning?')
    st.write('Universities may recognise dropout only after a student has disengaged or left. An early-warning estimate can help departments consider whether academic advising, financial-support review, study support, counselling, or a follow-up conversation may be useful. It does not automatically require intervention.')

def assessment(artifact:dict[str,Any])->None:
    st.title('Assess Dropout Risk'); st.caption("Complete this form using information from the student's most recent completed semester.")
    with st.form('assessment_form'):
        st.markdown('### Student Background'); c1,c2=st.columns(2)
        previous=c1.number_input('HSC / Equivalent GPA',2.50,5.00,3.50,step=0.01,help='Enter the student’s previous academic GPA on a 5.00 scale. The original training feature was linearly normalized to a 0–5 scale for interface simplicity. This aligns numerical scales only and does not imply equivalence between national grading systems.')
        age=c2.number_input('Age at Enrollment',17,70,19,help='Supported training range: 17–70 years.')
        st.markdown('### Family Background'); c1,c2=st.columns(2); labels=list(OCCUPATION_BY_LABEL)
        mother_label=c1.selectbox("Mother’s Occupation",labels,index=labels.index('Unskilled workers'))
        father_label=c2.selectbox("Father’s Occupation",labels,index=labels.index('Unskilled workers'))
        st.markdown('### Financial Status'); c1,c2,c3=st.columns(3)
        scholarship=c1.selectbox('Scholarship Holder',['No','Yes']); debtor=c2.selectbox('Debtor',['No','Yes']); tuition=c3.selectbox('Tuition Fees Up to Date',['No','Yes'])
        st.markdown('### Recent Semester Performance'); c1,c2,c3,c4=st.columns(4)
        enrolled=c1.number_input('Courses Enrolled',0,26,6); evaluations=c2.number_input('Evaluations Completed',0,45,8); approved=c3.number_input('Courses Passed',0,26,5); semester_gpa=c4.number_input('Semester GPA',0.00,4.00,3.00,step=0.01,help="Enter the student's most recently completed semester GPA on a 4.00 scale. For compatibility with the training dataset, the value is normalized internally to the model's original grade range.")
        submitted=st.form_submit_button('Estimate Dropout Risk',type='primary',width='stretch')
    if submitted:
        values={'previous_academic_gpa_normalized':previous,'Age at enrollment':age,"Mother's occupation":OCCUPATION_BY_LABEL[mother_label],"Father's occupation":OCCUPATION_BY_LABEL[father_label],'Scholarship holder':int(scholarship=='Yes'),'Debtor':int(debtor=='Yes'),'Tuition fees up to date':int(tuition=='Yes'),'Curricular units 1st sem (enrolled)':enrolled,'Curricular units 1st sem (evaluations)':evaluations,'Curricular units 1st sem (approved)':approved,'Curricular units 1st sem (grade)':semester_gpa_to_source_grade(semester_gpa)}
        errors=validate_inputs(values)
        if errors:
            for error in errors: st.error(error)
            return
        probability=dropout_probability(artifact,values); label=risk_label(probability); elevated=label.startswith('Elevated')
        st.markdown('### Dropout Early-Warning Assessment')
        st.markdown(f'<div class="result {"elevated" if elevated else "lower"}"><div>Estimated Dropout Risk</div><strong>{probability:.1%}</strong><h2>{label}</h2></div>',unsafe_allow_html=True)
        if elevated: st.write("The student's recent academic profile is similar to patterns associated with later dropout in the training data. Consider a supportive follow-up review.")
        else: st.write('No major immediate warning indicators were identified from the entered information. Continue normal academic monitoring.')
        st.warning('This is an estimated probability from an academic machine-learning model, not a certainty. The model has not been externally validated for Bangladeshi universities.')
        if elevated:
            indicators,supports=support_guidance(semester_gpa=semester_gpa,enrolled=enrolled,evaluations=evaluations,passed=approved,debtor=debtor=='Yes',tuition_up_to_date=tuition=='Yes',scholarship=scholarship=='Yes',previous_gpa=previous)
            st.markdown('#### Key Risk Indicators')
            st.caption('These indicators are based on the information entered and should not be interpreted as confirmed causes of dropout.')
            if indicators: st.markdown('\n'.join(f'- {indicator}' for indicator in indicators))
            else: st.write('No single actionable warning met the support-layer rule thresholds.')
            st.markdown('#### Possible Institute Support')
            if supports: st.markdown('\n'.join(f'- {support}' for support in supports))
            else: st.markdown('- Continue an individual academic-support review')

def about()->None:
    st.title('About Project')
    st.markdown('### Purpose'); st.write('This academic prototype demonstrates how machine learning can help identify students who may benefit from early support after a completed semester. It supports human review and does not make automatic academic decisions.')
    st.markdown('### Dataset and cohort'); st.write("The model uses the UCI Machine Learning Repository dataset “Predict Students’ Dropout and Academic Success.” Of 4,424 students, 3,630 had resolved outcomes: 2,209 graduates and 1,421 dropouts. The 794 still enrolled were excluded from supervised modeling.")
    st.markdown('### Model'); st.write('The Random Forest uses 11 inputs and a decision threshold of 0.48. Performance: 87.6% accuracy, 87.6% balanced accuracy, 81.9% precision, 87.7% recall, 84.7% F1, 93.8% ROC-AUC, and 92.9% PR-AUC.')
    st.markdown('### Methodology'); st.write("The source model was trained using first-semester academic performance. The prototype interface presents these fields as the student's most recently completed semester for easier local data collection. Local validation is required before operational use across different semester stages.")
    st.write('This is a numerical normalization for interface compatibility and does not imply equivalence between Portuguese and Bangladeshi grading systems.')
    st.markdown('### Local data collection plan'); st.write('A future Google Form can collect the same 11 user fields, plus research-only Current Semester / Academic Year and an Anonymous Follow-Up ID. Form responses without later outcome labels cannot immediately retrain a supervised model. Future actual outcomes must be linked using the anonymous ID before local validation or retraining.')
    st.write('The machine-learning model estimates dropout risk. A separate rule-based support layer summarizes relevant academic and financial warning indicators from the entered information and suggests possible institutional follow-up. These indicators are not causal explanations of dropout.')
    st.markdown('### Limitations'); st.markdown('- Training data originate from Portuguese higher education.\n- The model has not yet been externally validated on Bangladeshi university students.\n- The source training data use first-semester academic performance.\n- Local longitudinal outcome data are required before local retraining.\n- Predictions are statistical estimates, not certainties.\n- Outputs should support, not replace, human judgement.')
def style()->None:
    st.markdown('''<style>.block-container{max-width:1120px;padding-top:2.2rem;padding-bottom:3rem}h1{color:#18384f;letter-spacing:-.025em}h2,h3{color:#24566b}div[data-testid="stMetric"]{background:#f3f7f9;border:1px solid #d9e6ea;border-radius:14px;padding:1rem}.workflow{display:flex;flex-wrap:wrap;gap:.65rem;align-items:center;margin:1.5rem 0 2rem}.workflow span{background:#edf6f7;color:#174d5b;border:1px solid #cfe3e7;padding:.65rem .8rem;border-radius:10px;font-weight:650}.result{border-radius:16px;padding:1.25rem 1.5rem;margin:.5rem 0 1rem;border:1px solid}.result strong{font-size:2.6rem}.result h2{margin:.2rem 0}.result.elevated{background:#fff7ed;border-color:#f2c994}.result.lower{background:#edf8f2;border-color:#b9dfc8}@media(max-width:700px){.workflow b{display:none}}</style>''',unsafe_allow_html=True)

def main()->None:
    st.set_page_config(page_title='Student Dropout Early Warning System',page_icon='🎓',layout='wide'); style(); artifact=load_model_artifact()
    st.sidebar.title('Early Warning System'); page=st.sidebar.radio('Navigate',['Home','Assess Dropout Risk','About Project']); st.sidebar.caption('After a Completed Semester')
    if page=='Home': home()
    elif page=='Assess Dropout Risk': assessment(artifact)
    else: about()
if __name__=='__main__': main()