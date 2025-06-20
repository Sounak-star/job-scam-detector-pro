import os
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np

# Initialize NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# ===== DARK THEME =====
def set_dark_theme():
    st.markdown("""
    <style>
        :root {
            --primary: #3a86ff;
            --secondary: #8338ec;
            --accent: #3b82f6;
            --danger: #ff4d4d;
            --warning: #ffaa33;
            --success: #00cc88;
            --text: #ffffff;
            --text-light: #b3b3b3;
            --bg: #000000;
            --card-bg: #121212;
            --border: #333333;
        }
        
        body, .stApp {
            background-color: var(--bg);
            color: var(--text);
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }
        
        .header {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            padding: 2.5rem 2rem;
            border-radius: 0 0 20px 20px;
            margin-bottom: 2.5rem;
            box-shadow: 0 6px 25px rgba(0,0,0,0.3);
        }
        
        .card {
            background-color: var(--card-bg);
            border-radius: 12px;
            padding: 1.8rem;
            margin: 1.2rem 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            border: 1px solid var(--border);
            transition: all 0.3s ease;
        }
        
        .high-risk { 
            border-left: 4px solid var(--danger);
            background: linear-gradient(90deg, #1a1a1a 0%, #330000 100%);
        }
        .medium-risk { 
            border-left: 4px solid var(--warning);
            background: linear-gradient(90deg, #1a1a1a 0%, #332200 100%);
        }
        .low-risk { 
            border-left: 4px solid var(--success);
            background: linear-gradient(90deg, #1a1a1a 0%, #003322 100%);
        }
        
        .stTextArea textarea {
            background-color: var(--card-bg);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
            font-size: 15px;
            line-height: 1.6;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .stButton>button {
            background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 28px;
            font-weight: 600;
            font-size: 15px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(58, 134, 255, 0.3);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(58, 134, 255, 0.4);
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 1px solid var(--border);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: 500;
            transition: all 0.3s ease;
            color: var(--text-light);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
            color: white !important;
            box-shadow: 0 4px 12px rgba(58, 134, 255, 0.3);
        }
        
        .flag-item {
            display: flex;
            align-items: center;
            padding: 12px 16px;
            border-radius: 10px;
            background: #1a1a1a;
            margin: 8px 0;
            border-left: 3px solid var(--warning);
            transition: all 0.2s ease;
        }
        
        .flag-item:hover {
            transform: translateX(4px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .flag-item.critical {
            border-left: 3px solid var(--danger);
            background: #330000;
        }
        
        .risk-gauge {
            background: var(--card-bg);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.1);
            border: 1px solid var(--border);
        }
        
        .divider {
            border: 0;
            height: 1px;
            background: linear-gradient(90deg, rgba(58,134,255,0) 0%, rgba(58,134,255,0.3) 50%, rgba(58,134,255,0) 100%);
            margin: 2rem 0;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: var(--card-bg);
        }
        ::-webkit-scrollbar-thumb {
            background: var(--primary);
            border-radius: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

# ===== MODEL LOADING =====
@st.cache_resource
def load_model():
    # Enhanced training data with more scam examples
    train_data = [
        ("EARN $5,000/WEEK! No experience! Pay $99 fee", 1),
        ("Google work from home! $150 training fee", 1),
        ("Mystery shopper - deposit $200 first", 1),
        ("URGENT! $3,000/month - email@gmail.com", 1),
        ("Get paid to test apps! $50 verification", 1),
        ("Data entry clerks! No resume - pay $80", 1),
        ("Receive payments from clients and transfer them", 1),
        ("Must provide bank details for verification", 1),
        ("No experience needed—instant hiring!", 1),
        ("Software Engineer - 3+ years experience", 0),
        ("Marketing Intern - $20/hour", 0),
        ("Customer Service Rep - Full benefits", 0),
        ("Accountant - CPA required", 0),
        ("Remote Customer Support - $18/hour", 0)
    ]
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            token_pattern=r'\b[\w$]+\b',
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1500
        )),
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=0.9
        ))
    ])
    
    texts = [x[0] for x in train_data]
    labels = [x[1] for x in train_data]
    pipeline.fit(texts, labels)
    return pipeline

# ===== SCAM DETECTION =====
def detect_scam(text, model):
    RED_FLAGS = {
        r'\$\d{3,}': ("High salary claim", 0.8),
        'training fee': ("Upfront payment", 1.0),
        'sign up fee': ("Upfront payment", 1.0),
        'verification fee': ("Suspicious payment", 1.0),
        'no experience': ("Unrealistic requirements", 0.9),
        'no resume': ("Lack of professionalism", 0.8),
        '@gmail.com': ("Unprofessional email", 0.9),
        '@yahoo.com': ("Unprofessional email", 0.9),
        'refundable': ("Suspicious terms", 0.8),
        'limited positions': ("False urgency", 0.7),
        'apply now!': ("High pressure", 0.6),
        'google is hiring': ("Fake company", 1.0),
        'microsoft hiring': ("Fake company", 1.0),
        'bank details': ("Sensitive information request", 1.0),
        'transfer.*account': ("Money transfer request", 1.0),
        'instant hiring': ("Lack of process", 0.8),
        'verification purposes': ("Suspicious request", 0.9)
    }
    
    text = str(text).lower()
    text = re.sub(r'[^\w\s$@]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    found_flags = []
    for pattern, (desc, weight) in RED_FLAGS.items():
        if re.search(pattern, text):
            found_flags.append({
                'flag': re.search(pattern, text).group(),
                'description': desc,
                'weight': weight
            })
    
    # Calculate rule-based score
    rule_score = min(sum(f['weight'] for f in found_flags) / 3.0, 1.0)
    
    try:
        ml_score = model.predict_proba([text])[0][1]
    except:
        ml_score = 0.5
    
    # Combine scores with more weight on ML score
    if any(f['weight'] >= 0.9 for f in found_flags):
        # If critical flags found, ensure higher score
        final_score = max(ml_score, rule_score)
    else:
        final_score = 0.4 * rule_score + 0.6 * ml_score
    
    # Ensure score is between 0 and 1
    final_score = max(0.0, min(1.0, final_score))
    
    # Apply strict classification thresholds
    if final_score >= 0.7:
        verdict = "HIGH RISK"
    elif final_score >= 0.4:
        verdict = "MODERATE RISK"
    else:
        verdict = "LOW RISK"
    
    return {
        'score': final_score,
        'flags': found_flags,
        'verdict': verdict
    }

# ===== RISK GAUGE =====
def create_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score*100,
        domain={'x': [0, 1], 'y': [0, 1]},
        delta={'reference': 40},
        number={
            'suffix': "%",
            'font': {'size': 36, 'family': "Inter", 'color': "white"},
            'valueformat': ".1f"
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': "#b3b3b3",
                'tickfont': {'size': 12, 'color': "#b3b3b3"}
            },
            'bar': {'color': "#3a86ff", 'thickness': 0.25},
            'bgcolor': "#121212",
            'borderwidth': 0,
            'bordercolor': "#121212",
            'steps': [
                {'range': [0, 40], 'color': "#00cc88"},  # LOW RISK
                {'range': [40, 70], 'color': "#ffaa33"}, # MODERATE RISK
                {'range': [70, 100], 'color': "#ff4d4d"} # HIGH RISK
            ],
            'threshold': {
                'line': {'color': "#121212", 'width': 5},
                'thickness': 0.8,
                'value': score*100}
        }
    ))
    fig.update_layout(
        height=320,
        margin=dict(t=0, b=40, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter, Arial, sans-serif", 'color': "white"},
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ===== MAIN APP =====
def main():
    set_dark_theme()
    
    # Header
    st.markdown("""
    <div class="header">
        <h1 style='color: white; margin-bottom: 8px; font-size: 2.3rem; font-weight: 700;'>
        <span style="display: inline-block; margin-right: 12px;">🔍</span>Job Scam Detector PRO
        </h1>
        <p style='color: rgba(255,255,255,0.92); margin-bottom: 0; font-size: 1.05rem;'>
        AI-powered analysis of job postings for fraudulent indicators
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    model = load_model()
    
    # Main Tabs
    tab1, tab2 = st.tabs(["📋 Single Analysis", "📁 Batch Processing"])
    
    with tab1:
        col1, col2 = st.columns([2, 1], gap="large")
        
        with col1:
            st.markdown("### Job Description Analysis")
            job_desc = st.text_area(
                "Paste job posting here:",
                height=280,
                value="""Receive payments from clients and transfer them to company accounts.

Must provide bank details for 'verification purposes'.

No experience needed—instant hiring!

Contact: "Email: hiring@quickcashjobs.net\"""",
                help="Analyze any job posting for potential scam indicators",
                label_visibility="collapsed"
            )
            
            if st.button("Analyze Risk", type="primary", use_container_width=True):
                with st.spinner("Conducting comprehensive analysis..."):
                    result = detect_scam(job_desc, model)
                    st.session_state.result = result
                    st.session_state.processed = True
                    st.rerun()
        
        with col2:
            st.markdown("### Scam Indicators")
            with st.expander("Common red flags to watch for", expanded=True):
                st.markdown("""
                <div style="font-size: 15px; line-height: 1.7;">
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 20px; margin-right: 10px; color: #3a86ff;">💰</span>
                    <div><strong style="color: white;">Unrealistic salaries</strong><br>
                    <span style="color: #b3b3b3; font-size: 14px;">Claims of unusually high pay for minimal work</span></div>
                </div>
                
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 20px; margin-right: 10px; color: #3a86ff;">💸</span>
                    <div><strong style="color: white;">Upfront payments</strong><br>
                    <span style="color: #b3b3b3; font-size: 14px;">Requests for training fees or deposits</span></div>
                </div>
                
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 20px; margin-right: 10px; color: #3a86ff;">🏢</span>
                    <div><strong style="color: white;">Fake company claims</strong><br>
                    <span style="color: #b3b3b3; font-size: 14px;">Impersonating well-known companies</span></div>
                </div>
                
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 20px; margin-right: 10px; color: #3a86ff;">📧</span>
                    <div><strong style="color: white;">Unprofessional contacts</strong><br>
                    <span style="color: #b3b3b3; font-size: 14px;">Generic email domains instead of company emails</span></div>
                </div>
                
                <div style="display: flex; align-items: center; margin-top: 12px;">
                    <span style="font-size: 20px; margin-right: 10px; color: #3a86ff;">🔐</span>
                    <div><strong style="color: white;">Bank details requests</strong><br>
                    <span style="color: #b3b3b3; font-size: 14px;">Asking for sensitive financial information</span></div>
                </div>
                </div>
                """, unsafe_allow_html=True)
    
        if st.session_state.get('processed', False):
            result = st.session_state.result
            
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown("### Comprehensive Risk Report")
            
            col_a, col_b = st.columns([1, 1], gap="large")
            
            with col_a:
                st.markdown("#### Risk Assessment Score")
                st.plotly_chart(create_gauge(result['score']), use_container_width=True)
                
                # Use the verdict directly from the detection result
                risk_class = {
                    "HIGH RISK": "high-risk",
                    "MODERATE RISK": "medium-risk",
                    "LOW RISK": "low-risk"
                }.get(result['verdict'], "low-risk")
                
                risk_icon = {
                    "HIGH RISK": "🚨",
                    "MODERATE RISK": "⚠️",
                    "LOW RISK": "✅"
                }.get(result['verdict'], "✅")
                
                st.markdown(f"""
                <div class="card {risk_class}">
                    <div style="display: flex; align-items: center; margin-bottom: 12px;">
                        <span style="font-size: 28px; margin-right: 12px;">{risk_icon}</span>
                        <h3 style="margin: 0; color: white;">{result['verdict']}</h3>
                    </div>
                    <p style="color: #b3b3b3;">Confidence: <strong style="color: white;">{result['score']:.1%}</strong></p>
                    <p style="margin-top: 8px; font-size: 15px; color: #b3b3b3;">
                        { "This job posting shows strong indicators of fraudulent activity. Proceed with extreme caution." if result['verdict'] == "HIGH RISK" else 
                          "Suspicious elements detected. Requires thorough verification before proceeding." if result['verdict'] == "MODERATE RISK" else 
                          "Minimal risk indicators detected. Standard verification recommended." }
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                if result['flags']:
                    st.markdown("#### Detected Risk Factors")
                    
                    critical_flags = [f for f in result['flags'] if f['weight'] >= 0.8]
                    if critical_flags:
                        st.markdown("##### Critical Issues")
                        for flag in critical_flags:
                            st.markdown(f"""
                            <div class="flag-item critical">
                                <div style="font-size: 20px; margin-right: 12px; color: #ff4d4d;">⚠️</div>
                                <div>
                                    <strong style="color: #ff4d4d;">{flag['flag']}</strong>
                                    <div style="color: #b3b3b3; font-size: 14px; margin-top: 4px;">
                                        {flag['description']} • Severity: {flag['weight']:.0%}
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    other_flags = [f for f in result['flags'] if f['weight'] < 0.8]
                    if other_flags:
                        st.markdown("##### Other Considerations")
                        for flag in other_flags:
                            st.markdown(f"""
                            <div class="flag-item">
                                <div style="font-size: 20px; margin-right: 12px; color: #3a86ff;">ℹ️</div>
                                <div>
                                    <strong style="color: white;">{flag['flag']}</strong>
                                    <div style="color: #b3b3b3; font-size: 14px; margin-top: 4px;">
                                        {flag['description']} • Severity: {flag['weight']:.0%}
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("#### Keyword Analysis")
                try:
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='#121212',
                        colormap='Reds',
                        contour_width=1,
                        contour_color='#333333'
                    ).generate(job_desc)
                    plt.figure(figsize=(10,5), facecolor='#000000')
                    plt.imshow(wordcloud)
                    plt.axis("off")
                    st.pyplot(plt, use_container_width=True)
                except:
                    st.warning("Could not generate word cloud visualization")

    with tab2:
        st.markdown("### Batch Processing")
        
        st.markdown("**Don't have a CSV? Try this test data:**", help="Sample data to test the batch processing feature")
        if st.button("Generate Sample Jobs CSV"):
            test_data = {
                'title': [
                    "Google Remote Data Entry", 
                    "Amazon Customer Service",
                    "Mystery Shopper Needed",
                    "Payment Transfer Agent",
                    "Software Engineer Position"
                ],
                'description': [
                    "Earn $3,000/month! No experience. Pay $150 training fee. Email: googlejobs@gmail.com",
                    "Legit Amazon job. $18/hr. Apply at careers.amazon.com",
                    "Get paid to shop! Must deposit $200 first. Text 'SHOP' to 555-1234",
                    "Receive payments and transfer to company accounts. Provide bank details.",
                    "Senior developer needed. 5+ years Python. Competitive salary."
                ],
                'company': ["Google", "Amazon", "Shopping Co", "Quick Cash", "Tech Corp"]
            }
            st.session_state.test_csv = pd.DataFrame(test_data).to_csv(index=False).encode('utf-8')
        
        if 'test_csv' in st.session_state:
            st.download_button(
                label="Download Sample CSV",
                data=st.session_state.test_csv,
                file_name="sample_jobs.csv",
                mime="text/csv"
            )
        
        uploaded_file = st.file_uploader("Or upload your own CSV", type=["csv"], label_visibility="collapsed")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'description' not in df.columns:
                    st.error("Error: CSV must contain 'description' column")
                else:
                    with st.spinner("Analyzing jobs..."):
                        results = []
                        progress_bar = st.progress(0)
                        for i, row in df.iterrows():
                            results.append(detect_scam(row['description'], model))
                            progress_bar.progress((i + 1) / len(df))
                        
                        df['risk_score'] = [r['score'] for r in results]
                        df['risk_level'] = [r['verdict'] for r in results]
                        
                        st.dataframe(
                            df.sort_values('risk_score', ascending=False),
                            column_config={
                                "risk_score": st.column_config.ProgressColumn(
                                    "Risk Score",
                                    format="%.0f%%",
                                    min_value=0,
                                    max_value=1
                                )
                            },
                            height=400,
                            use_container_width=True
                        )
                        
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Results",
                            csv,
                            "job_scam_analysis.csv",
                            "text/csv"
                        )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Footer
    st.markdown("---")
    st.caption("ℹ️ Note: Always verify job postings through official company channels")

if __name__ == "__main__":
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'result' not in st.session_state:
        st.session_state.result = None
    main()
