import json
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import plotly.graph_objects as go

st.set_page_config(page_title="AI Traffic Analyzer", page_icon="🚦", layout="wide")

st.markdown("""
<style>
.gradient-text {
    background: -webkit-linear-gradient(45deg, #FF4B4B, #FF8F00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em;
    font-weight: 900;
    margin-bottom: 10px;
}
.metric-card {
    background-color: #262730;
    border-radius: 10px;
    padding: 20px;
    border: 1px solid #444;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

volume = ctrl.Antecedent(np.arange(0, 101, 1), 'volume')
weather = ctrl.Antecedent(np.arange(0, 11, 1), 'weather')
severity = ctrl.Consequent(np.arange(0, 101, 1), 'severity')

volume['low'] = fuzz.trimf(volume.universe, [0, 0, 50])
volume['medium'] = fuzz.trimf(volume.universe, [20, 50, 80])
volume['high'] = fuzz.trimf(volume.universe, [50, 100, 100])

weather['clear'] = fuzz.trimf(weather.universe, [0, 0, 5])
weather['bad'] = fuzz.trimf(weather.universe, [3, 10, 10])

severity['low'] = fuzz.trimf(severity.universe, [0, 0, 40])
severity['moderate'] = fuzz.trimf(severity.universe, [20, 50, 80])
severity['critical'] = fuzz.trimf(severity.universe, [60, 100, 100])

rule1 = ctrl.Rule(volume['high'] & weather['bad'], severity['critical'])
rule2 = ctrl.Rule(volume['high'] & weather['clear'], severity['moderate'])
rule3 = ctrl.Rule(volume['medium'], severity['moderate'])
rule4 = ctrl.Rule(volume['low'], severity['low'])

severity_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
severity_sim = ctrl.ControlSystemSimulation(severity_ctrl)

def calculate_severity(v_val, w_val):
    severity_sim.input['volume'] = v_val
    severity_sim.input['weather'] = w_val
    severity_sim.compute()
    return severity_sim.output['severity']

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google Gemini API Key", type="password")
    st.divider()
    st.write("Powered by **LangChain** and **scikit-fuzzy**")

st.markdown('<p class="gradient-text">Traffic Congestion Severity Analyzer</p>', unsafe_allow_html=True)
st.write("Measure live traffic density and weather impact using fuzzy logic and generative AI.")

user_input = st.text_area("Live Traffic Report:", placeholder="e.g., 'Torrential rain and cars are barely moving on the main highway...'")

if st.button("Analyze Severity", type="primary") and api_key and user_input:
    with st.spinner("Processing NLP and calculating fuzzy sets..."):
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0, google_api_key=api_key)
        output_parser = StrOutputParser()
        
        extract_prompt = PromptTemplate(
            input_variables=["text"],
            template="Extract traffic volume (0-100 scale) and weather severity (0-10 scale) from this text: '{text}'. Return ONLY a valid JSON object with keys 'volume' and 'weather'."
        )
        extract_chain = extract_prompt | llm | output_parser
        
        explain_prompt = PromptTemplate(
            input_variables=["severity_score", "text", "volume", "weather"],
            template="The user reported: '{text}'. The AI extracted volume={volume}/100 and weather={weather}/10. The fuzzy logic computed a congestion severity of {severity_score}/100. Write a conversational response explaining this severity and suggesting a basic action for drivers."
        )
        explain_chain = explain_prompt | llm | output_parser

        try:
            extracted_str = extract_chain.invoke({"text": user_input})
            
            cleaned_str = extracted_str.strip()
            if cleaned_str.startswith("```json"):
                cleaned_str = cleaned_str[7:-3]
            elif cleaned_str.startswith("```"):
                cleaned_str = cleaned_str[3:-3]
                
            data = json.loads(cleaned_str)
            v = int(data.get("volume", 50))
            w = int(data.get("weather", 0))
            
            score = calculate_severity(v, w)
            explanation = explain_chain.invoke({
                "severity_score": round(score, 1), 
                "text": user_input, 
                "volume": v, 
                "weather": w
            })
            
            st.divider()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Overall Congestion", 'font': {'size': 20, 'color': "white"}},
                    number = {'font': {'color': "white"}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "white", 'thickness': 0.15},
                        'bgcolor': "#1E1E1E",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 40], 'color': "#00C864"},
                            {'range': [40, 70], 'color': "#FFC800"},
                            {'range': [70, 100], 'color': "#FF3232"}]}))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f'<div class="metric-card"><p style="color:#aaa;margin:0;">Traffic Volume</p><h2 style="margin:0;">{v} / 100</h2></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card"><p style="color:#aaa;margin:0;">Weather Severity</p><h2 style="margin:0;">{w} / 10</h2></div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.info(explanation)
            
        except Exception as e:
            st.error(f"Error processing the input: {e}")