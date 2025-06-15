import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib


# Load all the required models
@st.cache_resource
def load_model():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("noc_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    return vectorizer, model, label_encoder

df_top_k_noc_list = pd.read_csv('./noc_code_2021_mapper.csv')
top_k_noc_code_descr_map = {}

for noc_code, desc in zip(df_top_k_noc_list['noc_code'],df_top_k_noc_list['noc_title']):
    top_k_noc_code_descr_map[noc_code] = desc


def plot_predictions(noc_codes, confidences):
    colors = []
    for score in confidences:
        if score >= 0.80:
            colors.append("#2ECC71")  # Green
        elif score >= 0.60:
            colors.append("#F39C12")  # Gold/Orange
        else:
            colors.append("#0A66C2")  # Blue (default)

    fig = go.Figure(
        data=[
            go.Bar(
                x=noc_codes,
                y=confidences,
                marker_color=colors,
                text=[f"{c*100:.2f}%" for c in confidences],
                textposition='auto'
            )
        ]
    )
    fig.update_layout(
        title="Top 3 Predicted NOC Codes",
        xaxis_title="NOC Code",
        yaxis_title="Confidence",
        yaxis_range=[0, 1],
        height=400
    )
    return fig

# Page configuration
st.set_page_config(
    page_title="NOC Code Predictor",
    page_icon= "🍁",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #F5F8FA;
        }
        .main-title {
            font-size: 36px;
            font-weight: 700;
            color: #00416A;
            text-align: center;
        }
        .subtitle {
            font-size: 18px;
            color: #222831;
            text-align: center;
            margin-bottom: 30px;
        }
        .info-box {
            background-color: #e3f2fd;
            color: #00416A;
            padding: 15px;
            border-radius: 10px;
            border-left: 6px solid #00A3E0;
            font-size: 15px;
        }
        .confidence-high {
            color: white;
            background-color: #4CAF50;
            padding: 3px 10px;
            border-radius: 5px;
            font-weight: 600;                     
        }
        .confidence-medium {
            color: black;
            background-color: #FFC107;
            padding: 3px 10px;
            border-radius: 5px;
            font-weight: 600;            
        }
        .confidence-low {
            color: white;
            background-color: #F44336;
            padding: 3px 10px;
            border-radius: 5px;
            font-weight: 600;            
        }
        
        
    </style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown('<div class="main-title">🔍 Canadian NOC Code Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict the most suitable NOC (National Occupation Classification) code based on your job title and description.</div>', unsafe_allow_html=True)

# Info message
st.markdown("""
<div class="info-box">
    💡 <strong>How it works:</strong> Enter a job title and description. Our AI model will predict the best matching NOC code, with confidence scores and alternate suggestions.</strong>.
</div>
""", unsafe_allow_html=True)

st.write("")

# Function to determine confidence class
def get_confidence_class(indx):
    if indx == 1:
        return "confidence-high"
    elif indx == 2:
        return "confidence-medium"
    else:
        return "confidence-low"
    
supported_noc_code_list = ""

#st.write(top_k_noc_code_descr_map)

for noc_code in top_k_noc_code_descr_map:
    supported_noc_code_list = supported_noc_code_list + "- NOC " + str(noc_code).zfill(5) + " - " + top_k_noc_code_descr_map[noc_code] + "\n"

def truncate_string(text, max_length=30):
  if len(text) <= max_length:    
    return text
  else:
    return text[:max_length] + "..."

jobData = {}

if 'job_title' not in st.session_state:
    st.session_state.job_title = ""

if 'job_description' not in st.session_state:
    st.session_state.job_description = ""

def OnSubmitFormClear():
    st.session_state.job_title = ""
    st.session_state.job_description = ""


with st.form("job_data_form"):
    
    # User input
    job_title = st.text_input("Job Title*", key= "job_title", placeholder="e.g. Software Engineer")   

    st.markdown("""
        <div style="background-color:#e7f3fe; padding:10px 15px; border-left: 6px solid #2196F3; border-radius:6px; color:#084298;">
            💡 <strong>Tip:</strong> Enter a proper and detailed job description for the best prediction results.
        </div>
        """, unsafe_allow_html=True)

    job_description  = st.text_area("Job Description / Resume*", key="job_description", height=200, placeholder="Paste your job responsibilities, tools, or keywords...")

    submitBtn, clearBtn = st.columns([1,1])

    with submitBtn:
        submitResult = st.form_submit_button("🔎 Predict NOC Code")

    with clearBtn:
        clearResult = st.form_submit_button("🧹 Clear",on_click=OnSubmitFormClear)
   
    
    if clearResult:
        st.write("")

    if submitResult:
               
        jobData['job_title'] = job_title.strip()
        jobData['job_details'] = job_description.strip()       
            
        if job_title.strip() and job_description.strip():                        

            vectorizer, model, label_encoder = load_model()
            combine_text = jobData['job_title'] + " " + jobData['job_details']
            X = vectorizer.transform([combine_text])
            y_proba = model.predict_proba(X)[0]

            top_indices = y_proba.argsort()[-3:][::-1]
            top_nocs = label_encoder.inverse_transform(top_indices)
            top_scores = y_proba[top_indices]

            st.markdown("""
                <div style="background-color:#fff3cd; padding:15px 20px; border-left: 6px solid #ffecb5; border-radius:8px; color:#856404;">
                    <strong>⚠️ Note:</strong> This model currently supports predictions for the <strong>top 100 most in-demand NOC codes</strong> in Canada.  
                    You may not receive accurate results for less common job types or improper job description.<br>
                    We’re working on improving and expanding coverage to include all NOC codes soon.
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            
            st.markdown(f"#### 🏁 Predicted {top_nocs[0]} – {truncate_string(top_k_noc_code_descr_map[(int)(top_nocs[0][4:])]).title()}")
            
            noc_code_pred_results = []            

            noc_code_predictions = zip(top_nocs, top_scores)    

            for noc, score in noc_code_predictions:
                noc_code_pred_result = {}
                md_str = f"{noc} - {top_k_noc_code_descr_map[(int)(noc[4:])]} - {(score * 100)}"
                noc_code_pred_result['noc'] = noc
                noc_code_pred_result['title'] = top_k_noc_code_descr_map[(int)(noc[4:])]
                noc_code_pred_result['confidence'] = score * 100                
                noc_code_pred_results.append(noc_code_pred_result)
            
            # Generate markdown HTML
            markdown_lines = []
            for idx, item in enumerate(noc_code_pred_results, start=1):                
                conf_percent = item["confidence"]
                confidence_class = get_confidence_class(idx)
                line = f"{idx}. {item['noc']} – {truncate_string(item['title'],60).title()} <span class='{confidence_class}'>{conf_percent:.2f} %</span><br>"
                markdown_lines.append(line)         

            markdown_result = "\n".join(markdown_lines)

            st.markdown("#### 🏆 Top 3 Predictions")
                #st.markdown(markdown_result, unsafe_allow_html=True)

            table_html = """
                <table style="width:100%; border-collapse: separate; border-spacing: 1px 1px;">
                <thead>
                <tr style="text-align:left;">
                    <th width="20%">NOC Code</th>
                    <th>Job Title</th>
                    <th style="text-align:right;">Confidence</th>
                </tr>
                </thead>
                <tbody>
                """
            indx = 0
            for noc, score in zip(top_nocs, top_scores):
                indx = indx + 1
                title = truncate_string(top_k_noc_code_descr_map[int(noc[4:])],50).title()
                conf = f"{score * 100:.2f}%"
                confidence_class = get_confidence_class(indx)
                table_html += f"""
                <tr>
                    <td>{noc}</td>
                    <td>{title}</td>
                    <td class='{confidence_class}'><strong>{conf}</strong></td>
                </tr>
                """

            table_html += "</tbody></table>"

            st.markdown(table_html, unsafe_allow_html=True)

            fig = plot_predictions(top_nocs, top_scores)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 View Supported NOC Codes (Top 100)"):
                st.markdown(f"""
                    { supported_noc_code_list }
                    """)           
        
        else:
            st.error("⚠️ Please fill out both the Job Title and Job Description fields.")


    
                    