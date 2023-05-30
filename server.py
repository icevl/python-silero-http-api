from langdetect import detect
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn
import torchaudio
import tempfile
import os
import torch

app = FastAPI()

device = torch.device('cpu')
torch.set_num_threads(4)

local_file_ru = 'model_ru.pt'
local_file_en = 'model_en.pt'

if not os.path.isfile(local_file_ru):
    torch.hub.download_url_to_file(
        'https://models.silero.ai/models/tts/ru/v3_1_ru.pt', local_file_ru)

if not os.path.isfile(local_file_en):
    torch.hub.download_url_to_file(
        'https://models.silero.ai/models/tts/en/v3_en.pt', local_file_en)

model_ru = torch.package.PackageImporter(
    local_file_ru).load_pickle("tts_models", "model")
model_en = torch.package.PackageImporter(
    local_file_en).load_pickle("tts_models", "model")

model_ru.to(device)
model_en.to(device)


class TextPayload(BaseModel):
    text: str


def get_language_model(text):
    lang = detect(text)
    if lang == "en":
        return model_en, "en_0"
    return model_ru, "baya"


@app.post("/tts")
def tts_test(item: TextPayload):
    text_cleared = item.text.replace('\n', '').replace('\r', '')

    language_model = get_language_model(text_cleared)
    model, speaker = language_model
    sample_rate = 24000

    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filepath = temp_file.name

            audio = model.apply_tts(
                text=text_cleared, speaker=speaker, sample_rate=sample_rate)
            if audio.dim() != 2:
                audio = audio.unsqueeze(0)

            torchaudio.save(temp_filepath, audio, sample_rate,
                            encoding="PCM_S", bits_per_sample=16)

            with open(temp_filepath, 'rb') as audio_file:
                response = Response(content=audio_file.read())
                response.headers["Content-Type"] = "audio/wav"
                os.remove(temp_filepath)
                return response

    except Exception as e:
        print(f"An error occurred: {e}")
        os.remove(temp_filepath)
        return


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
