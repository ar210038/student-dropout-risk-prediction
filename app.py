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
RESULTS_PATH=ROOT/'reports'/'uci_sem1'/'training_results_11_input.json'
CM_PATH=ROOT/'reports'/'uci_sem1'/'figures'/'confusion_matrix_11_input.png'
CALIBRATION_PATH=ROOT/'reports'/'uci_sem1'/'figures'/'calibration_curve_11_input.png'
THRESHOLD=0.48
SOURCE_SEMESTER_GRADE_MAX=18.875
FORM_FIELDS=['previous_academic_gpa_normalized','Age at enrollment',"Mother's occupation","Father's occupation",'Scholarship holder','Debtor','Tuition fees up to date','Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)','Curricular units 1st sem (approved)','Curricular units 1st sem (grade)']
OCCUPATIONS={
0:'Student',1:'Legislative/executive representatives, directors and managers',2:'Intellectual and scientific specialists',3:'Intermediate-level technicians and professions',4:'Administrative staff',5:'Personal services, security and sales workers',6:'Skilled agriculture, fisheries and forestry workers',7:'Skilled industry, construction and craft workers',8:'Machine operators and assembly workers',9:'Unskilled workers',10:'Armed Forces',90:'Other situation',99:'Unspecified',101:'Armed Forces officers',102:'Armed Forces sergeants',103:'Other Armed Forces personnel',112:'Administrative and commercial services directors',114:'Hotel, catering, trade and other service directors',121:'Physical sciences, mathematics and engineering specialists',122:'Health professionals',123:'Teachers',124:'Finance, administration and commercial specialists',125:'Information and communication technology specialists',131:'Intermediate science and engineering technicians',132:'Intermediate health technicians',134:'Intermediate legal, social, cultural and sports services',135:'Information and communication technology technicians',141:'Office, secretarial and data-processing workers',143:'Data, accounting, statistics and financial operators',144:'Other administrative support staff',151:'Personal service workers',152:'Sellers',153:'Personal care workers',154:'Protection and security personnel',161:'Market-oriented farmers and skilled agricultural workers',163:'Subsistence farmers, fishers, hunters and gatherers',171:'Skilled construction workers (except electricians)',172:'Skilled metallurgy and metalworking workers',173:'Printing, precision, jewellery and craft workers',174:'Skilled electrical and electronics workers',175:'Food, wood, clothing and other craft workers',181:'Fixed-plant and machine operators',182:'Assembly workers',183:'Vehicle drivers and mobile-equipment operators',191:'Cleaning workers',192:'Unskilled agriculture, fisheries and forestry workers',193:'Unskilled extractive, construction, manufacturing and transport workers',194:'Meal preparation assistants',195:'Street vendors and service providers'}
OCCUPATION_BY_LABEL={label:code for code,label in OCCUPATIONS.items()}
FEATURE_IMPORTANCE=pd.DataFrame({'Feature':['Courses Passed','Courses Enrolled / Pass Rate','Tuition Fees Up to Date','Semester-1 GPA','Scholarship','Father’s Occupation','Age','Mother’s Occupation','Evaluations Completed','HSC / Equivalent GPA','Debtor Status'],'Association':[.1433,.0339,.0262,.0216,.0164,.0093,.0090,.0089,.0083,.0081,.0072]})

@st.cache_resource
def load_model_artifact()->dict[str,Any]: return joblib.load(MODEL_PATH)
@st.cache_data
def load_metadata()->tuple[dict,dict]: return json.loads(METADATA_PATH.read_text(encoding='utf-8')),json.loads(RESULTS_PATH.read_text(encoding='utf-8'))

def make_input_frame(values:dict[str,Any])->pd.DataFrame:
    if set(values)!=set(FORM_FIELDS): raise ValueError('Answers do not match the required 12-field form.')
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

def cards(items:list[tuple[str,str]])->None:
    for column,(label,value) in zip(st.columns(len(items)),items): column.metric(label,value)

def home()->None:
    st.title('Student Dropout Early Warning System')
    st.subheader('Machine Learning-Based Early Identification After the First Semester')
    st.write("This academic prototype estimates a student's risk of later university dropout using information available after the first semester. The goal is to help identify students who may benefit from earlier academic or financial support.")
    st.markdown('<div class="workflow"><span>Semester 1 Completed</span><b>→</b><span>Student Information</span><b>→</b><span>Machine Learning</span><b>→</b><span>Dropout Risk Estimate</span><b>→</b><span>Early Support</span></div>',unsafe_allow_html=True)
    cards([('Prediction point','End of Semester 1'),('Model','Random Forest'),('Training cohort','3,630 resolved outcomes'),('Inputs','11'),('Test ROC-AUC','93.8%')])
    st.markdown('### Why Early Warning?')
    st.write('Universities may recognise dropout only after a student has disengaged or left. An early-warning estimate can help departments consider whether academic advising, financial-support review, study support, counselling, or a follow-up conversation may be useful. It does not automatically require intervention.')

def assessment(artifact:dict[str,Any])->None:
    st.title('Assess Dropout Risk'); st.caption('Complete this form after the student has finished Semester 1.')
    with st.form('assessment_form'):
        st.markdown('### Student Background'); c1,c2=st.columns(2)
        previous=c1.number_input('HSC / Equivalent GPA',2.50,5.00,3.50,step=0.01,help='Enter the student’s previous academic GPA on a 5.00 scale. The original training feature was linearly normalized to a 0–5 scale for interface simplicity. This aligns numerical scales only and does not imply equivalence between national grading systems.')
        age=c2.number_input('Age at Enrollment',17,70,19,help='Supported training range: 17–70 years.')
        st.markdown('### Family Background'); c1,c2=st.columns(2); labels=list(OCCUPATION_BY_LABEL)
        mother_label=c1.selectbox("Mother’s Occupation",labels,index=labels.index('Unskilled workers'))
        father_label=c2.selectbox("Father’s Occupation",labels,index=labels.index('Unskilled workers'))
        st.markdown('### Financial Status'); c1,c2,c3=st.columns(3)
        scholarship=c1.selectbox('Scholarship Holder',['No','Yes']); debtor=c2.selectbox('Debtor',['No','Yes']); tuition=c3.selectbox('Tuition Fees Up to Date',['No','Yes'])
        st.markdown('### First-Semester Performance'); c1,c2,c3,c4=st.columns(4)
        enrolled=c1.number_input('Courses Enrolled',0,26,6); evaluations=c2.number_input('Evaluations Completed',0,45,8); approved=c3.number_input('Courses Passed',0,26,5); semester_gpa=c4.number_input('Semester-1 GPA',0.00,4.00,3.00,step=0.01,help="Enter the student's first-semester GPA on a 4.00 scale. For compatibility with the training dataset, the value is normalized internally to the model's original Semester-1 grade range.")
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
        if elevated: st.write("The student's first-semester profile is similar to patterns associated with later dropout in the model's training data. This result may be used as an early signal for further academic or student-support review.")
        else: st.write("The student's first-semester profile is associated with a lower estimated risk of later dropout in the training data. This does not guarantee that the student will remain enrolled.")
        st.warning('This is an estimated probability from an academic machine-learning model, not a certainty. Probability calibration was usable in the held-out evaluation but has not been externally validated for Bangladeshi universities.')
        if elevated:
            st.markdown('#### Possible Follow-Up'); st.markdown('- Academic advising\n- Review of Semester-1 difficulties\n- Financial-support discussion where relevant\n- Student-services or counselling referral where appropriate\n- Follow-up conversation with the student')

def performance(results:dict)->None:
    st.title('Model Performance'); st.write('Locked Random Forest evaluation on the untouched 20% holdout set.')
    cards([('Threshold','0.48'),('Accuracy','87.6%'),('Balanced Accuracy','87.6%'),('Precision','81.9%'),('Recall','87.7%')]); cards([('F1','84.7%'),('ROC-AUC','93.8%'),('PR-AUC','92.9%'),('Brier Score','0.0997'),('Alert rate','41.87%')])
    st.markdown('### What the metrics mean'); st.markdown('- **Recall:** Among held-out students who later dropped out, approximately 88% were identified.\n- **Precision:** Among students flagged as elevated risk, approximately 82% were dropout cases in the held-out dataset.\n- **F1:** Balances precision and recall.\n- **ROC-AUC:** Measures separation of dropout and graduate outcomes across thresholds.\n- **PR-AUC:** Measures precision-recall performance and is useful when classes are not perfectly balanced.')
    st.markdown('### Confusion matrix'); cards([('Correctly identified graduates','387'),('False early warnings','55'),('Missed dropout cases','35'),('Correctly identified dropout cases','249')]); st.image(str(CM_PATH),width='stretch')
    st.markdown("### Features associated with the model's dropout predictions")
    chart=FEATURE_IMPORTANCE.sort_values('Association'); fig,ax=plt.subplots(figsize=(8,5)); ax.barh(chart['Feature'],chart['Association'],color='#266b7a'); ax.set_xlabel('Training permutation importance (ROC-AUC decrease)'); ax.spines[['top','right']].set_visible(False); fig.tight_layout(); st.pyplot(fig,width='stretch'); plt.close(fig)
    st.caption('Importance reflects predictive association in this model and does not establish causation. Pass rate is derived from Courses Passed and Courses Enrolled.')
    st.markdown('### Probability calibration'); st.image(str(CALIBRATION_PATH),width='stretch'); st.caption('Calibration was usable on the held-out dataset, with some variation across probability ranges. Local recalibration may be needed.')

def about()->None:
    st.title('About & Methodology')
    st.markdown('### Dataset'); st.write("The model uses the UCI Machine Learning Repository dataset “Predict Students’ Dropout and Academic Success.”")
    cards([('Original students','4,424'),('Resolved cohort','3,630'),('Graduate','2,209'),('Dropout','1,421'),('Enrolled excluded','794')])
    st.markdown('### Why Enrolled was excluded'); st.write('Students labelled Enrolled did not yet have a resolved final outcome. Treating them as dropout or non-dropout would create an incorrect target, so only Graduate and Dropout outcomes were used.')
    st.markdown('### Prediction timing and leakage controls'); st.write('The prediction point is the **end of the first semester**. Background, financial, and Semester-1 performance information is allowed. Target, every Semester-2 variable, final-outcome information, Application Mode, Application Order, Course code, Nationality, unemployment, inflation, and GDP are excluded. **No information occurring after the defined prediction point is used by the final model.**')
    st.markdown('### Simplified form and normalized academic result'); st.write("The final system uses 11 user inputs. The original dataset records the student's previous qualification grade on a different numerical scale. For interface simplicity, this feature was linearly normalized during model training using `source grade / 190 × 5`, producing the observed 2.50–5.00 range.")
    st.warning('This normalization aligns numerical scales only; it does not imply that grading systems from different countries are academically equivalent.')
    st.write("The visible Semester-1 GPA is entered on a 0–4 scale and normalized internally to the source model's observed 0–18.875 grade range. This is a numerical normalization for interface compatibility and does not imply equivalence between Portuguese and Bangladeshi grading systems.")
    st.write('Compared with the validated 12-input Random Forest, repeated-CV ROC-AUC changed by −0.0001, PR-AUC by −0.0005, recall by −0.0004, and F1 by −0.0016. The smaller form was selected to improve usability while preserving essentially all cross-validation performance.')
    st.markdown('### Limitations'); st.markdown('- Dataset comes from Portuguese higher education\n- Model has not been validated on Bangladeshi university students\n- Academic systems may differ between countries\n- Occupation categories originate from the source dataset\n- Probability calibration may require local recalibration\n- Predictions are statistical estimates, not certainties\n- Outputs should support, not replace, human judgement\n- Local longitudinal data is needed before operational deployment')
    st.info('This system is an academic prototype. Although the interface is designed to be understandable in a general university setting, the model was trained using Portuguese higher-education data. Operational use in Bangladesh would require external validation or retraining using local longitudinal student records.')

def style()->None:
    st.markdown('''<style>.block-container{max-width:1120px;padding-top:2.2rem;padding-bottom:3rem}h1{color:#18384f;letter-spacing:-.025em}h2,h3{color:#24566b}div[data-testid="stMetric"]{background:#f3f7f9;border:1px solid #d9e6ea;border-radius:14px;padding:1rem}.workflow{display:flex;flex-wrap:wrap;gap:.65rem;align-items:center;margin:1.5rem 0 2rem}.workflow span{background:#edf6f7;color:#174d5b;border:1px solid #cfe3e7;padding:.65rem .8rem;border-radius:10px;font-weight:650}.result{border-radius:16px;padding:1.25rem 1.5rem;margin:.5rem 0 1rem;border:1px solid}.result strong{font-size:2.6rem}.result h2{margin:.2rem 0}.result.elevated{background:#fff7ed;border-color:#f2c994}.result.lower{background:#edf8f2;border-color:#b9dfc8}@media(max-width:700px){.workflow b{display:none}}</style>''',unsafe_allow_html=True)

def main()->None:
    st.set_page_config(page_title='Student Dropout Early Warning System',page_icon='🎓',layout='wide'); style(); artifact=load_model_artifact(); _,results=load_metadata()
    st.sidebar.title('Early Warning System'); page=st.sidebar.radio('Navigate',['Home','Assess Dropout Risk','Model Performance','About & Methodology']); st.sidebar.caption('Prediction point · End of Semester 1')
    if page=='Home': home()
    elif page=='Assess Dropout Risk': assessment(artifact)
    elif page=='Model Performance': performance(results)
    else: about()
if __name__=='__main__': main()
