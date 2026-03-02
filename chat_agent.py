import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"  # Use phi for faster response


class ChatAgent:

    def __init__(self):
        pass

    def build_prompt(self, user_question, result, chat_history):
        condition = result["predicted_condition"]
        confidence = result["confidence_percentage"]

        # Format previous conversation
        history_text = ""
        for item in chat_history:
            history_text += f"User: {item['question']}\n"
            history_text += f"Assistant: {item['answer']}\n"

        prompt = f"""
You are a professional AI medical assistant supporting hospital staff.

A lung X-ray was analyzed using a deep learning model.

===============================
PATIENT DIAGNOSIS CONTEXT
===============================
Detected Condition: {condition}
Model Confidence: {confidence}%

===============================
PREVIOUS CONVERSATION
===============================
{history_text}

===============================
NEW USER QUESTION
===============================
{user_question}

===============================
INSTRUCTIONS
===============================
- Answer clearly and professionally.
- Use medical reasoning based on the detected condition.
- Refer to previous conversation if relevant.
- Do NOT hallucinate unknown clinical facts.
- If uncertainty exists, state it clearly.
- Provide helpful guidance but avoid making definitive medical decisions.
- Always remind that final decisions must be made by a licensed physician.

End your response with:
"This analysis is AI-assisted and must be verified by a licensed medical professional."
"""

        return prompt

    def call_ollama(self, prompt):

        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3
            }
        }

        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"LLM Error: {response.status_code} | {response.text}"

        except Exception as e:
            return f"Connection Error: {str(e)}"

    def ask(self, user_question, result, chat_history):

        prompt = self.build_prompt(user_question, result, chat_history)
        answer = self.call_ollama(prompt)

        return answer