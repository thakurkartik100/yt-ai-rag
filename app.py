import gradio as gr
from app.main import app

# Hide the Gradio interface inside the mount call so Hugging Face doesn't try to launch it instead of our API
app = gr.mount_gradio_app(
    app, 
    gr.Interface(
        fn=lambda: "The Tube AI RAG API is running! Append /docs to the URL above to see the Swagger UI.", 
        inputs=None, 
        outputs="text",
        title="Tube AI RAG API"
    ), 
    path="/"
)
