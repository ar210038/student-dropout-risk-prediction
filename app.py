"""Local Student Dropout Early Warning System."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import io
import joblib
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
MODEL_PATH=ROOT/'models'/'uci_sem1'/'dropout_early_warning_model.joblib'
METADATA_PATH=ROOT/'models'/'uci_sem1'/'model_metadata.json'
THRESHOLD=0.48
SOURCE_SEMESTER_GRADE_MAX=18.875
FORM_FIELDS=['previous_academic_gpa_normalized','Age at enrollment',"Mother's occupation","Father's occupation",'Scholarship holder','Debtor','Tuition fees up to date','Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)','Curricular units 1st sem (approved)','Curricular units 1st sem (grade)']
USER_FIELDS=['HSC / Equivalent GPA','Age at Enrollment',"Mother's Occupation","Father's Occupation",'Scholarship Holder','Debtor','Tuition Fees Up to Date','Courses Enrolled','Evaluations Completed','Courses Passed','Semester GPA']
ID_COLUMN='Anonymous Follow-Up ID'
SEMESTER_COLUMN='Current Semester / Academic Year'
BATCH_COLUMNS=[ID_COLUMN,SEMESTER_COLUMN,*USER_FIELDS]
RESULT_COLUMNS=[ID_COLUMN,SEMESTER_COLUMN,'Estimated Dropout Risk','Risk Level','Main Indicator','Key Risk Indicators','Suggested Institute Support']
OCCUPATIONS={0:'Student',1:'Legislative/executive representatives, directors and managers',2:'Intellectual and scientific specialists',3:'Intermediate-level technicians and professions',4:'Administrative staff',5:'Personal services, security and sales workers',6:'Skilled agriculture, fisheries and forestry workers',7:'Skilled industry, construction and craft workers',8:'Machine operators and assembly workers',9:'Unskilled workers',10:'Armed Forces',90:'Other situation',99:'Unspecified',101:'Armed Forces officers',102:'Armed Forces sergeants',103:'Other Armed Forces personnel',112:'Administrative and commercial services directors',114:'Hotel, catering, trade and other service directors',121:'Physical sciences, mathematics and engineering specialists',122:'Health professionals',123:'Teachers',124:'Finance, administration and commercial specialists',125:'Information and communication technology specialists',131:'Intermediate science and engineering technicians',132:'Intermediate health technicians',134:'Intermediate legal, social, cultural and sports services',135:'Information and communication technology technicians',141:'Office, secretarial and data-processing workers',143:'Data, accounting, statistics and financial operators',144:'Other administrative support staff',151:'Personal service workers',152:'Sellers',153:'Personal care workers',154:'Protection and security personnel',161:'Market-oriented farmers and skilled agricultural workers',163:'Subsistence farmers, fishers, hunters and gatherers',171:'Skilled construction workers (except electricians)',172:'Skilled metallurgy and metalworking workers',173:'Printing, precision, jewellery and craft workers',174:'Skilled electrical and electronics workers',175:'Food, wood, clothing and other craft workers',181:'Fixed-plant and machine operators',182:'Assembly workers',183:'Vehicle drivers and mobile-equipment operators',191:'Cleaning workers',192:'Unskilled agriculture, fisheries and forestry workers',193:'Unskilled extractive, construction, manufacturing and transport workers',194:'Meal preparation assistants',195:'Street vendors and service providers'}
OCCUPATION_BY_LABEL={label:code for code,label in OCCUPATIONS.items()}

@st.cache_resource
def load_model_artifact()->dict[str,Any]: return joblib.load(MODEL_PATH)

def semester_gpa_to_source_grade(value:float)->float: return float(value)/4.0*SOURCE_SEMESTER_GRADE_MAX

def make_input_frame(values:dict[str,Any])->pd.DataFrame:
    if set(values)!=set(FORM_FIELDS): raise ValueError('Answers do not match the required 11-field form.')
    return pd.DataFrame([{field:values[field] for field in FORM_FIELDS}])

def validate_inputs(values:dict[str,Any])->list[str]:
    return ['Courses Passed cannot be greater than Courses Enrolled.'] if values['Curricular units 1st sem (approved)']>values['Curricular units 1st sem (enrolled)'] else []

def dropout_probability(artifact:dict[str,Any],values:dict[str,Any])->float:
    pipeline=artifact['pipeline']; index=list(pipeline.classes_).index(1)
    return float(pipeline.predict_proba(make_input_frame(values))[0,index])

def risk_label(probability:float)->str: return 'Elevated Estimated Risk' if probability>=THRESHOLD else 'Lower Estimated Risk'

def support_guidance(*,semester_gpa:float,enrolled:int,evaluations:int,passed:int,debtor:bool,tuition_up_to_date:bool,scholarship:bool,previous_gpa:float)->tuple[list[str],list[str]]:
    pass_rate=passed/enrolled if enrolled else 0.0; evaluation_rate=evaluations/enrolled if enrolled else 0.0; candidates=[]
    if semester_gpa<2.0: candidates.append((10,f'Low recent academic performance — Semester GPA: {semester_gpa:.2f} / 4.00',['Academic advising','Tutoring / study support']))
    if not enrolled: candidates.append((9,'No course enrollment was reported',['Study-plan review']))
    elif pass_rate<0.5: candidates.append((9,f'Low course completion — {passed} of {enrolled} courses passed',['Course-specific academic support','Study-plan review']))
    elif pass_rate<0.7: candidates.append((5,f'Moderate course completion — {passed} of {enrolled} courses passed',['Study-plan review']))
    if not tuition_up_to_date: candidates.append((8,'Tuition fees are not currently up to date',['Financial-aid review','Installment/payment support if available']))
    if debtor: candidates.append((8,'Outstanding debtor status was reported',['Financial counselling','Review possible payment arrangements']))
    if enrolled and evaluation_rate<0.75: candidates.append((7,f'Low evaluation completion — {evaluations} evaluations for {enrolled} enrolled courses',['Student engagement follow-up','Check attendance / participation difficulties']))
    if not scholarship and (debtor or not tuition_up_to_date): candidates.append((4,'No scholarship was reported alongside a financial warning',['Review scholarship or student-aid eligibility']))
    if previous_gpa<3.0: candidates.append((3,f'Weaker previous academic background — HSC / Equivalent GPA: {previous_gpa:.2f} / 5.00',['Foundation/remedial academic support if needed']))
    selected=sorted(candidates,key=lambda item:item[0],reverse=True)[:3]; supports=[]
    for _,_,recommendations in selected:
        if recommendations[0] not in supports: supports.append(recommendations[0])
    for _,_,recommendations in selected:
        for recommendation in recommendations[1:]:
            if recommendation not in supports and len(supports)<4: supports.append(recommendation)
    return [item[1] for item in selected],supports

def parse_boolean(value:Any)->bool:
    if isinstance(value,bool): return value
    if isinstance(value,(int,float)) and not pd.isna(value) and float(value) in (0,1): return bool(value)
    normalized=str(value).strip().lower()
    if normalized in {'yes','true','1'}: return True
    if normalized in {'no','false','0'}: return False
    raise ValueError('must be Yes/No, TRUE/FALSE, or 1/0')

def occupation_code(value:Any)->int:
    if value in OCCUPATION_BY_LABEL: return OCCUPATION_BY_LABEL[value]
    try: code=int(float(value))
    except (TypeError,ValueError): raise ValueError('is not a supported occupation label')
    if code not in OCCUPATIONS: raise ValueError('is not a supported occupation label')
    return code

def prepare_student(record:dict[str,Any])->dict[str,Any]:
    return {'previous_academic_gpa_normalized':float(record['HSC / Equivalent GPA']),'Age at enrollment':int(record['Age at Enrollment']),"Mother's occupation":occupation_code(record["Mother's Occupation"]),"Father's occupation":occupation_code(record["Father's Occupation"]),'Scholarship holder':int(parse_boolean(record['Scholarship Holder'])),'Debtor':int(parse_boolean(record['Debtor'])),'Tuition fees up to date':int(parse_boolean(record['Tuition Fees Up to Date'])),'Curricular units 1st sem (enrolled)':int(record['Courses Enrolled']),'Curricular units 1st sem (evaluations)':int(record['Evaluations Completed']),'Curricular units 1st sem (approved)':int(record['Courses Passed']),'Curricular units 1st sem (grade)':semester_gpa_to_source_grade(float(record['Semester GPA']))}

def predict_student(artifact:dict[str,Any],record:dict[str,Any])->dict[str,Any]:
    values=prepare_student(record); probability=dropout_probability(artifact,values); label=risk_label(probability)
    indicators,supports=support_guidance(semester_gpa=float(record['Semester GPA']),enrolled=int(record['Courses Enrolled']),evaluations=int(record['Evaluations Completed']),passed=int(record['Courses Passed']),debtor=bool(values['Debtor']),tuition_up_to_date=bool(values['Tuition fees up to date']),scholarship=bool(values['Scholarship holder']),previous_gpa=float(record['HSC / Equivalent GPA']))
    return {'probability':probability,'risk_level':label,'indicators':indicators,'supports':supports}

def validate_batch_dataframe(frame:pd.DataFrame)->pd.DataFrame:
    missing=[column for column in USER_FIELDS if column not in frame.columns]
    if missing: return pd.DataFrame([{'Row':'File','Issue':'Missing required columns: '+', '.join(missing)}])
    issues=[]
    for position,(_,row) in enumerate(frame.iterrows(),start=2):
        def issue(message:str): issues.append({'Row':position,'Issue':message})
        try:
            hsc=float(row['HSC / Equivalent GPA'])
            if not 2.5<=hsc<=5.0: issue('HSC / Equivalent GPA must be between 2.50 and 5.00.')
        except (TypeError,ValueError): issue('HSC / Equivalent GPA must be numeric.')
        try:
            gpa=float(row['Semester GPA'])
            if not 0<=gpa<=4: issue('Semester GPA must be between 0.00 and 4.00.')
        except (TypeError,ValueError): issue('Semester GPA must be numeric.')
        for column,minimum,maximum in [('Age at Enrollment',17,70),('Courses Enrolled',0,26),('Evaluations Completed',0,None),('Courses Passed',0,26)]:
            try:
                value=float(row[column])
                if value<minimum or (maximum is not None and value>maximum) or not value.is_integer(): issue(f'{column} must be a whole number in the supported range.')
            except (TypeError,ValueError): issue(f'{column} must be numeric.')
        try:
            if float(row['Courses Passed'])>float(row['Courses Enrolled']): issue('Courses Passed cannot be greater than Courses Enrolled.')
        except (TypeError,ValueError): pass
        for column in ['Scholarship Holder','Debtor','Tuition Fees Up to Date']:
            try: parse_boolean(row[column])
            except ValueError as error: issue(f'{column} {error}.')
        for column in ["Mother's Occupation","Father's Occupation"]:
            try: occupation_code(row[column])
            except ValueError as error: issue(f'{column} {error}.')
    return pd.DataFrame(issues,columns=['Row','Issue'])

def indicator_group(indicator:str)->str:
    return 'Financial' if any(word in indicator.lower() for word in ['tuition','debtor','scholarship']) else 'Academic'

def metadata_text(value:Any,fallback:str='')->str:
    return fallback if value is None or pd.isna(value) or not str(value).strip() else str(value).strip()

def analyze_batch(artifact:dict[str,Any],frame:pd.DataFrame)->pd.DataFrame:
    errors=validate_batch_dataframe(frame)
    if not errors.empty: raise ValueError(errors.to_json(orient='records'))
    rows=[]
    for position,(_,source) in enumerate(frame.iterrows(),start=1):
        record=source.to_dict(); result=predict_student(artifact,record); indicators=result['indicators'] if result['risk_level'].startswith('Elevated') else []
        rows.append({**record,ID_COLUMN:metadata_text(record.get(ID_COLUMN),f'Student {position}'),SEMESTER_COLUMN:metadata_text(record.get(SEMESTER_COLUMN)), 'Estimated Dropout Risk':result['probability'],'Risk Level':result['risk_level'],'Main Indicator':indicators[0] if indicators else '-','Key Risk Indicators':' | '.join(indicators) if indicators else '-','Suggested Institute Support':' | '.join(result['supports']) if indicators else '-','Warning Type':indicator_group(indicators[0]) if indicators else '-'})
    return pd.DataFrame(rows)

def export_results(frame:pd.DataFrame)->bytes:
    columns=[*BATCH_COLUMNS,*RESULT_COLUMNS[2:]]
    return frame[[column for column in columns if column in frame.columns]].to_csv(index=False).encode('utf-8')

def template_csv()->bytes:
    example={ID_COLUMN:'EXAMPLE-001',SEMESTER_COLUMN:'3rd Semester','HSC / Equivalent GPA':4.20,'Age at Enrollment':19,"Mother's Occupation":'Administrative staff',"Father's Occupation":'Unskilled workers','Scholarship Holder':'No','Debtor':'No','Tuition Fees Up to Date':'Yes','Courses Enrolled':6,'Evaluations Completed':6,'Courses Passed':5,'Semester GPA':3.25}
    return pd.DataFrame([example],columns=BATCH_COLUMNS).to_csv(index=False).encode('utf-8')

def cards(items:list[tuple[str,str]])->None:
    for column,(value,label) in zip(st.columns(len(items)),items):
        column.markdown(f'<div class="metric-card"><strong>{value}</strong><span>{label}</span></div>',unsafe_allow_html=True)

def navigate(page:str)->None: st.session_state['page_selector']=page

def dashboard()->None:
    st.title('Student Dropout Early Warning System'); st.subheader('Machine Learning-Based Early Identification for Student Support')
    st.write('Identify potentially at-risk students and support them earlier using academic, financial, and background information.')
    cards([('Random Forest','ML Model'),('11','Student Factors'),('93.8%','ROC-AUC'),('87.7%','Recall')])
    st.markdown('<div class="workflow"><span>Student Data</span><b>→</b><span>Risk Analysis</span><b>→</b><span>Warning Indicators</span><b>→</b><span>Early Support</span></div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    c1.button('Assess One Student',type='primary',use_container_width=True,key='dashboard_assess_one',on_click=navigate,args=('Individual Assessment',))
    c2.button('Analyze a Student Batch',use_container_width=True,key='dashboard_analyze_batch',on_click=navigate,args=('Batch Analysis',))

def render_result(result:dict[str,Any],student_id:str='Individual Assessment')->None:
    elevated=result['risk_level'].startswith('Elevated')
    st.markdown(f'<div class="result {"elevated" if elevated else "lower"}"><span>Estimated Dropout Risk</span><strong>{result["probability"]:.1%}</strong><h2>{result["risk_level"].upper()}</h2></div>',unsafe_allow_html=True)
    if elevated:
        st.markdown('#### Key Risk Indicators'); st.caption('These indicators are based on the entered information and are not confirmed causes of dropout.')
        st.markdown('\n'.join(f'- {item}' for item in result['indicators']) if result['indicators'] else '- No single actionable warning met the rule thresholds.')
        st.markdown('#### Possible Institute Support'); st.markdown('\n'.join(f'- {item}' for item in result['supports']) if result['supports'] else '- Continue an individual academic-support review')
    else: st.success('No major immediate warning indicators were identified from the entered information. Continue normal academic monitoring.')
    report='\n'.join([f'Student: {student_id}',f'Estimated Dropout Risk: {result["probability"]:.1%}',f'Risk Level: {result["risk_level"]}','Key Risk Indicators: '+('; '.join(result['indicators']) or 'None'),'Suggested Institute Support: '+('; '.join(result['supports']) or 'Normal academic monitoring')])
    st.download_button('Download Assessment',report,file_name='student_assessment.txt',mime='text/plain')

def individual_assessment(artifact:dict[str,Any])->None:
    st.title('Individual Assessment'); st.caption("Use information from the student's most recent completed semester.")
    with st.form('assessment_form'):
        student_id=st.text_input('Anonymous Student ID (optional)')
        st.markdown('### Student Background'); c1,c2=st.columns(2); previous=c1.number_input('HSC / Equivalent GPA',2.50,5.00,3.50,step=0.01); age=c2.number_input('Age at Enrollment',17,70,19)
        st.markdown('### Family Background'); c1,c2=st.columns(2); labels=list(OCCUPATION_BY_LABEL); mother=c1.selectbox("Mother’s Occupation",labels,index=labels.index('Unskilled workers')); father=c2.selectbox("Father’s Occupation",labels,index=labels.index('Unskilled workers'))
        st.markdown('### Financial Status'); c1,c2,c3=st.columns(3); scholarship=c1.selectbox('Scholarship Holder',['No','Yes']); debtor=c2.selectbox('Debtor',['No','Yes']); tuition=c3.selectbox('Tuition Fees Up to Date',['No','Yes'])
        st.markdown('### Recent Semester Performance'); c1,c2,c3,c4=st.columns(4); enrolled=c1.number_input('Courses Enrolled',0,26,6); evaluations=c2.number_input('Evaluations Completed',0,45,8); passed=c3.number_input('Courses Passed',0,26,5); semester_gpa=c4.number_input('Semester GPA',0.00,4.00,3.00,step=0.01)
        submitted=st.form_submit_button('Estimate Dropout Risk',type='primary',use_container_width=True)
    if submitted:
        record={'HSC / Equivalent GPA':previous,'Age at Enrollment':age,"Mother's Occupation":mother,"Father's Occupation":father,'Scholarship Holder':scholarship,'Debtor':debtor,'Tuition Fees Up to Date':tuition,'Courses Enrolled':enrolled,'Evaluations Completed':evaluations,'Courses Passed':passed,'Semester GPA':semester_gpa}
        if passed>enrolled: st.error('Courses Passed cannot be greater than Courses Enrolled.'); return
        render_result(predict_student(artifact,record),student_id.strip() or 'Individual Assessment')

def batch_analysis(artifact:dict[str,Any])->None:
    st.title('Batch Analysis'); st.write('Upload a Google Forms/Sheets CSV to assess multiple students with the same model and threshold.')
    st.info('For demonstration and research use, avoid uploading names, phone numbers, email addresses, or other unnecessary personally identifiable information. Use anonymous student IDs.')
    st.download_button('Download CSV Template',template_csv(),'student_batch_template.csv','text/csv')
    with st.expander('Using Google Forms'):
        st.markdown('1. Create a Google Form with the same student questions.\n2. Store responses in Google Sheets.\n3. Export the Sheet as CSV.\n4. Upload the CSV here.\n5. Analyze all students together.\n\nDirect Google Sheets synchronization can be considered as future work.')
    uploaded=st.file_uploader('Upload student CSV',type='csv')
    if uploaded is None: return
    try: frame=pd.read_csv(uploaded)
    except Exception as error: st.error(f'Could not read CSV: {error}'); return
    errors=validate_batch_dataframe(frame)
    if not errors.empty:
        st.error('The file contains validation problems. No rows were analyzed.'); st.dataframe(errors,use_container_width=True,hide_index=True); return
    results=analyze_batch(artifact,frame); elevated=results['Risk Level'].str.startswith('Elevated')
    cards([(str(len(results)),'Total Students'),(str(int(elevated.sum())),'Elevated Risk'),(str(int((~elevated).sum())),'Lower Risk'),(f'{results["Estimated Dropout Risk"].mean():.1%}','Average Estimated Risk')])
    st.markdown('### Risk Distribution'); distribution=results['Risk Level'].value_counts().rename_axis('Risk Level').to_frame('Students'); st.bar_chart(distribution)
    elevated_indicators=results.loc[elevated,'Key Risk Indicators'].str.split(' | ',regex=False).explode(); elevated_indicators=elevated_indicators[elevated_indicators!='-']
    st.markdown('### Most Common Warning Indicators'); st.bar_chart(elevated_indicators.value_counts().head(8).rename_axis('Indicator').to_frame('Students'))
    st.markdown('### Student Results'); c1,c2,c3=st.columns(3); risk_filter=c1.selectbox('Risk Level',['All','Elevated','Lower']); semesters=['All']+sorted(x for x in results[SEMESTER_COLUMN].dropna().astype(str).unique() if x); semester_filter=c2.selectbox('Semester',semesters); search=c3.text_input('Search Anonymous Follow-Up ID')
    filtered=results
    if risk_filter!='All': filtered=filtered[filtered['Risk Level'].str.startswith(risk_filter)]
    if semester_filter!='All': filtered=filtered[filtered[SEMESTER_COLUMN].astype(str)==semester_filter]
    if search: filtered=filtered[filtered[ID_COLUMN].astype(str).str.contains(search,case=False,na=False)]
    display=filtered[[ID_COLUMN,SEMESTER_COLUMN,'Estimated Dropout Risk','Risk Level','Main Indicator']].copy(); display['Estimated Dropout Risk']=display['Estimated Dropout Risk'].map(lambda x:f'{x:.1%}'); st.dataframe(display,use_container_width=True,hide_index=True)
    st.download_button('Download Analysis Results CSV',export_results(results),'student_batch_analysis.csv','text/csv')

def about()->None:
    st.title('About Project')
    st.markdown('### Purpose'); st.write('An early-warning support tool for identifying students who may benefit from timely academic or financial follow-up.')
    st.markdown('### Dataset'); st.write("UCI “Predict Students’ Dropout and Academic Success”: 4,424 original records; 3,630 resolved outcomes used (2,209 graduate, 1,421 dropout); 794 enrolled records excluded.")
    st.markdown('### Model'); st.write('Random Forest using 11 student factors and a decision threshold of 0.48.')
    st.markdown('### Performance'); st.write('Accuracy 87.6% · Precision 81.9% · Recall 87.7% · F1 84.7% · ROC-AUC 93.8% · PR-AUC 92.9%')
    st.markdown('### How Risk Indicators Work'); st.write('The machine-learning model estimates dropout risk. A separate rule-based support layer summarizes relevant academic and financial warning indicators and suggests possible institutional follow-up. These indicators are not causal explanations of dropout.')
    st.markdown('### Local Data Collection'); st.write('A campus Google Form can collect the same student information used by the system. These records can be used to study local patterns. Once actual academic outcomes are later linked to anonymous records, the data may be used for local validation and future model retraining.')
    st.markdown('### Limitations'); st.markdown('- Training data originate from Portuguese higher education and have not been externally validated in Bangladesh.\n- Source academic variables describe first-semester performance; broader semester use requires local validation.\n- Local longitudinal outcomes are required before retraining.\n- Predictions are estimates and should support, not replace, human judgement.')

def style()->None:
    st.markdown('''<style>:root{--navy:#102f4a;--blue:#1f6485}.stApp{background:#f7f9fb}.block-container{max-width:1180px;padding-top:2rem;padding-bottom:3rem}h1{color:var(--navy);letter-spacing:-.035em}h2,h3,h4{color:#194b68}.metric-card{background:white;border:1px solid #e0e8ee;border-radius:16px;padding:1.2rem;box-shadow:0 5px 18px rgba(16,47,74,.07);min-height:105px}.metric-card strong{display:block;color:var(--navy);font-size:1.75rem}.metric-card span{color:#627485}.workflow{display:flex;flex-wrap:wrap;gap:.7rem;align-items:center;margin:1.6rem 0}.workflow span{background:white;color:#174d69;border:1px solid #d7e4eb;padding:.75rem 1rem;border-radius:12px;font-weight:650;box-shadow:0 3px 12px rgba(16,47,74,.05)}.result{border-radius:18px;padding:1.4rem 1.6rem;margin:1rem 0;border:1px solid;box-shadow:0 5px 18px rgba(16,47,74,.07)}.result strong{display:block;font-size:2.8rem}.result h2{margin:.15rem 0}.result.elevated{background:#fff6ee;border-color:#efb47c}.result.elevated h2{color:#a53b21}.result.lower{background:#eef9f2;border-color:#addbbd}.result.lower h2{color:#176b3a}div[data-testid="stForm"],div[data-testid="stExpander"]{background:white;border-radius:16px;border-color:#e0e8ee}@media(max-width:700px){.workflow b{display:none}}</style>''',unsafe_allow_html=True)

def main()->None:
    st.set_page_config(page_title='Student Dropout Early Warning System',page_icon='🎓',layout='wide'); style(); artifact=load_model_artifact()
    pages=['Dashboard','Individual Assessment','Batch Analysis','About Project']; st.sidebar.title('Early Warning System'); page=st.sidebar.radio('Navigate',pages,key='page_selector'); st.sidebar.caption('Local academic prototype')
    if page=='Dashboard': dashboard()
    elif page=='Individual Assessment': individual_assessment(artifact)
    elif page=='Batch Analysis': batch_analysis(artifact)
    else: about()
if __name__=='__main__': main()
