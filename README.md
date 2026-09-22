# 🚦 AI-Based Traffic Congestion Severity Analyzer

An interactive web application that combines the natural language processing capabilities of Generative AI with the mathematical determinism of a Fuzzy Inference System to evaluate traffic congestion based on unstructured citizen reports. This project bridges modern IT with the multi-valued logic principles of the Indian Knowledge System (IKS).

**Student Details:**
* **Name:** Pondharani Devendra
* **Roll Number:** 19010
* **Class:** TY IT 
* **Subject:** (Indian Knowledge System)

**Live Deployment Link:** [Traffic Analyzer Web App](https://ai-fuzzy-traffic-dzeqwzd2tkjvm5trdjdwzd.streamlit.app/)

## 📸 Screenshot
<img width="624" height="371" alt="image" src="https://github.com/user-attachments/assets/ce2c3591-2b71-457e-9f6f-8d0dd88629b3" />

## 🌟 Main Features
* **Natural Language Parsing:** Extracts numerical variables (Traffic Volume and Weather) from unstructured conversational text using LangChain and Google Gemini.
* **Fuzzy Logic Evaluation:** Calculates a deterministic severity score (0-100) using `scikit-fuzzy` triangular membership functions and Mamdani rule evaluation.
* **AI-Synthesized Advisory:** Generates a context-aware recommendation for drivers based on the mathematical severity score.
* **Dynamic Dashboard:** Visualizes data using Plotly gauge charts and metric cards within a Streamlit UI.

## 💻 Technologies & Tech Stack
* **Language:** Python 3.11+
* **Frontend:** Streamlit, Custom CSS
* **Generative AI:** LangChain Core, LangChain Google GenAI, Google Gemini 3.6-flash
* **Mathematics & Logic:** `scikit-fuzzy`, NumPy, SciPy
* **Data Visualization:** Plotly Graph Objects

## 🔐 Environment Variables
To run this project, you need a valid Google Gemini API key. Never expose your actual API key in the code.
* `GOOGLE_API_KEY`: Enter this directly into the application's sidebar interface when running the app.

## 🚀 Installation and Setup

1. **Clone the Repository**
```bash
git clone [https://github.com/Dharani-dev22/ai-fuzzy-traffic.git](https://github.com/Dharani-dev22/ai-fuzzy-traffic.git)
cd ai-fuzzy-traffic

2. **Install Dependencies**

    pip install -r requirements.txt

3. **Run the Application**

    streamlit run app.py


## 📦 Requirements (requirements.txt)

    streamlit
    langchain
    langchain-google-genai
    langchain-core
    scikit-fuzzy
    scipy
    numpy
    networkx
    plotly
