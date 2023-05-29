from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from langdetect import detect
import threading
import torchaudio
import tempfile
import os
import torch
import json

http_port = 10000

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


def get_language_model(text):
    lang = detect(text)
    if lang == "en":
        return model_en, "en_0"
    return model_ru, "baya"


class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        form_data = self.rfile.read(content_length)

        try:
            json_data = json.loads(form_data)
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Invalid JSON data')
            return

        if 'text' in json_data:
            text = json_data['text'].replace('\n', '').replace('\r', '')

            language_model = get_language_model(text)
            model, speaker = language_model
            sample_rate = 24000

            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_filepath = temp_file.name
                    audio = model.apply_tts(
                        text=text, speaker=speaker, sample_rate=sample_rate)
                    if audio.dim() != 2:
                        audio = audio.unsqueeze(0)

                    torchaudio.save(temp_filepath, audio, sample_rate,
                                    encoding="PCM_S", bits_per_sample=16)

                    self.send_response(200)
                    self.send_header('Content-type', 'audio/wav')
                    self.end_headers()

                    with open(temp_filepath, 'rb') as audio_file:
                        self.wfile.write(audio_file.read())

                os.remove(temp_filepath)
            except:
                self.send_response(400)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Invalid model response')
                return

        else:
            self.send_response(400)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Missing "text" parameter')


server_address = ('', http_port)
httpd = ThreadingHTTPServer(server_address, RequestHandler)

server_thread = threading.Thread(target=httpd.serve_forever)
# server_thread.daemon = True
server_thread.start()
print("*Server started at port:", http_port)
