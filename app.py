import gradio as gr
from sentence_transformers import SentenceTransformer
from PIL import Image
import fitz  # PyMuPDF for PDFs
from groq import Groq
import pytesseract

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Groq client
groq_api_key = "secret_Api_key"  # Replace with your actual key
client = Groq(api_key=groq_api_key)

# System + user prompt templates
SYSTEM_TEMPLATE = "You are a professional psychological specialist. Help the patient understand their condition and provide a 7-day recovery or support plan."
USER_TEMPLATE = """The patient shared the following information:

{context}

The patient’s question or concern:
{question}

Give a detailed and kind response along with a 7-day improvement plan.
"""

# Extract text from PDF or image
def extract_text(file):
    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    elif file.name.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(file.name)
        return image_to_text(image)
    return ""

# OCR for image
def image_to_text(image):
    return pytesseract.image_to_string(image)

# Chunk and embed
def chunk_and_embed(text):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    embeddings = embedding_model.encode(chunks)
    return list(zip(chunks, embeddings))

# Cosine similarity
def cosine_similarity(a, b):
    return sum(ai * bi for ai, bi in zip(a, b)) / ((sum(ai**2 for ai in a) ** 0.5) * (sum(bi**2 for bi in b) ** 0.5) + 1e-8)

# Retrieve relevant context
def retrieve_context(query, chunks, k=2):
    query_vec = embedding_model.encode(query)
    ranked = sorted(chunks, key=lambda x: -cosine_similarity(x[1], query_vec))
    return "\n\n".join([c[0] for c in ranked[:k]])

# Main handler
def handle_interaction(user_input, file):
    extracted_text = ""
    if file is not None:
        extracted_text = extract_text(file)
    chunks = chunk_and_embed(extracted_text) if extracted_text else []
    context = retrieve_context(user_input, chunks) if chunks else "No document provided."

    final_prompt = USER_TEMPLATE.format(context=context, question=user_input)

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user", "content": final_prompt}
        ],
        model="llama3-8b-8192"
    )
    return chat_completion.choices[0].message.content.strip()

# Custom CSS
css = """
#app-wrapper {
    background-image: url("https://static.vecteezy.com/system/resources/thumbnails/027/003/992/small/green-natural-leaves-background-free-photo.jpg");
    background-size: full;
    background-position: center;
    padding: 60px;
    min-height: 100vh;
}

.container-box {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 20px;
    padding: 20px;
    color: white;
}
.container-box1 {
    background-color: black;
    border-radius: 20px;
    box-border=round;
    padding: 20px;
    color:transparent;
}
textarea, input, .gr-textbox {
    background-color: transparent !important;
    color: white !important;
    border: none !important;
    box-shadow: none !important;
}

.gr-textbox textarea {
    background-color: black !important;
    color: white !important;
}

.gr-button {
    background-color: rgba(255, 255, 255, 0.1) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 12px;
}
"""

# Gradio Interface
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="app-wrapper"):
        gr.Markdown("# 🧠 Psychological RAG Assistant", elem_classes="container-box")

        with gr.Row():
            with gr.Column():
                user_input = gr.Textbox(
                    label="📝 Describe your mental health concern or ask a question",
                    placeholder="Type your concern or query...",
                    lines=10,
                    elem_classes="container-box1"
                )
            with gr.Column():
                file_upload = gr.File(label="📎 Upload psychological report (PDF/Image)", file_types=[".pdf", ".png", ".jpg", ".jpeg"], elem_classes="container-box")

        submit = gr.Button("💬 Get Solution & 7-Day Plan", elem_classes="container-box")

        output = gr.Textbox(label="🧠 AI Response", lines=10, elem_classes="container-box1")

        submit.click(fn=handle_interaction, inputs=[user_input, file_upload], outputs=output)

demo.launch()
