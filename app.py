import gradio as gr
from app.main import app as fastapi_app

# 1. Create a dummy Gradio UI just to satisfy Hugging Face's requirements
def ui_message():
    return "The Tube AI RAG API is running! Append /docs to the URL above to see the Swagger UI."

demo = gr.Interface(
    fn=ui_message, 
    inputs=None, 
    outputs="text",
    title="Tube AI RAG API",
    description="This is a headless API. Go to the /docs endpoint to use it."
)

# 2. Mount the Gradio UI onto our existing FastAPI app
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
