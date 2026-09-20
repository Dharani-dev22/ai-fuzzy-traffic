import json
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

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

st.set_page_config(page_title="AI Traffic Analyzer")
st.title("Traffic Congestion Severity Analyzer")

api_key = st.text_input("Google Gemini API Key", type="password")
user_input = st.text_area("Enter traffic report (e.g., 'Heavy rain and cars are stuck on the bridge'):")

if st.button("Analyze") and api_key and user_input:
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
        
        st.subheader("System Outputs")
        st.write(f"**Extracted Variables (LangChain):** Volume: {v}/100, Weather: {w}/10")
        st.write(f"**Computed Severity (Fuzzy Logic):** {score:.1f} / 100")
        st.write("**Conversational Explanation (LangChain):**")
        st.write(explanation)
        
    except Exception as e:
        st.error(f"Error processing the input: {e}")