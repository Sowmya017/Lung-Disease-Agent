import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "mistral"

class ChatAgent:
    def __init__(self):
        self.conversation_history = []
        self.current_context = None

    def ask(self, question, context):
        # Update context if new result comes in
        if context != self.current_context:
            self.current_context = context
            self.conversation_history = []  # reset history for new scan

        system_message = {
            "role": "system",
            "content": (
                f"You are a helpful medical assistant. "
                f"A lung X-ray was analyzed with the following result:\n"
                f"Detected Condition: {context['predicted_condition']}\n"
                f"Model Confidence: {context['confidence_percentage']}%\n\n"
                f"Answer the user's questions clearly and helpfully. "
                f"Always remind users to consult a licensed doctor for medical decisions."
            )
        }

        self.conversation_history.append({
            "role": "user",
            "content": question
        })

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "messages": [system_message] + self.conversation_history,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            answer = response.json()["message"]["content"]

        except requests.exceptions.ConnectionError:
            answer = "❌ Ollama is not running. Please start it with: ollama serve"
        except Exception as e:
            answer = f"❌ Error: {str(e)}"

        self.conversation_history.append({
            "role": "assistant",
            "content": answer
        })

        return answer