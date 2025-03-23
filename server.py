from fastapi import FastAPI, UploadFile, File
import whisper, openai
from pythonosc.udp_client import SimpleUDPClient
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()
model = whisper.load_model("base")
openai.api_key = os.getenv("OPENAI_API_KEY")
osc_client = SimpleUDPClient("127.0.0.1", 5005)  # Update if needed

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    with open("temp.wav", "wb") as f:
        f.write(await file.read())

    result = model.transcribe("temp.wav")
    text = result["text"]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": text}]
    )
    gpt_output = response["choices"][0]["message"]["content"]

    osc_client.send_message("/chat_output", gpt_output)

    return {"transcription": text, "gpt_output": gpt_output}
