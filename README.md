# 🚦 AI-Based Traffic Congestion Severity Analyzer

A Streamlit web application that bridges the gap between unstructured human language and deterministic mathematical modeling. This project uses generative AI to extract quantifiable variables from raw citizen traffic reports, applies a Fuzzy Inference System to calculate a severity score, and synthesizes a conversational advisory for drivers.

## Live link: [Traffic Analyzer Streamlit App](https://ai-fuzzy-traffic-dzeqwzd2tkjvm5trdjdwzd.streamlit.app/)

## Screenshot:
<img width="624" height="371" alt="image" src="https://github.com/user-attachments/assets/ce2c3591-2b71-457e-9f6f-8d0dd88629b3" />

## 🌟 Features

*   **Natural Language Processing:** Uses Google's `gemini-3.6-flash` via LangChain to parse messy, real-world text into strict JSON integers.
*   **Fuzzy Logic Engine:** Implements `scikit-fuzzy` with triangular membership functions and four overlapping rules to compute a deterministic traffic congestion score (0-100).
*   **Conversational Advisory:** Leverages a secondary LangChain sequence to translate the mathematical output into a context-aware recommendation.
*   **Professional Dashboard:** Built with Streamlit, custom CSS, and dynamic Plotly gauge charts for a premium UI experience.

## 🚀 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Dharani-dev22/ai-fuzzy-traffic.git](https://github.com/Dharani-dev22/ai-fuzzy-traffic.git)
    cd ai-fuzzy-traffic
    ```

2.  **Install Dependencies**
    Ensure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

4.  **Configure API Key**
    Once the app launches in your browser at `localhost:8501`, paste your free Google Gemini API key into the sidebar to start analyzing traffic reports.

## 📦 Requirements (`requirements.txt`)

```text
streamlit
langchain
langchain-google-genai
langchain-core
scikit-fuzzy
scipy
numpy
networkx
plotly
