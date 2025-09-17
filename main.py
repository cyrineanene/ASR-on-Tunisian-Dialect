import gradio as gr
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor
from peft import PeftModel

# --- Configuration ---
CHECKPOINT_PATH = "all_checkpoints/checkpoints7_12/lora_checkpoints/lora_model_epoch_8_step_2500"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Base Whisper + LoRA Adapters ---
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
base_model.to(DEVICE)
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="ar", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small", language="ar", task="transcribe")

try:
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.to(DEVICE)
    print("✅ LoRA adapters loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading LoRA adapters: {e}")
    print("Proceeding with base Whisper model only.")
    model = base_model 

model.eval()

# --- Transcription Function ---
def transcribe_audio(mic_input, file_input):
    audio_path = mic_input if mic_input else file_input
    if not audio_path:
        return "❌ No audio provided"

    try:
        # Load audio with librosa
        audio_data, sr = librosa.load(audio_path, sr=16000)

        # Extract features
        input_features = feature_extractor(
            raw_speech=audio_data,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(DEVICE)

        # 👇 NEW: get forced decoder IDs for language + task
        forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="arabic", task="transcribe")

        # Run inference
        with torch.no_grad():
            generated_ids = model.generate(
                input_features=input_features,
                forced_decoder_ids=forced_decoder_ids,  # ✅ replaces language/task
                max_new_tokens=140
            )

        # Decode transcription
        transcription = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # return 'حتلقاو بلاد دخلوها يتفرجو فنهوجتها فسوارها منها شدو لثنية روحو دارهم'
        return transcription

    except Exception as e:
        return f"⚠️ Error processing audio: {str(e)}"


# --- Gradio UI ---
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            record = gr.Audio(sources=["microphone"], type="filepath", label="record")
            filein = gr.Audio(sources=["upload"], type="filepath", label="filein")
            
            with gr.Row():
                clear_btn = gr.ClearButton([record, filein], value="Nettoyer")
                submit_btn = gr.Button("Soumettre")

        with gr.Column(scale=1):
            output = gr.Textbox(label="output", lines=10, max_lines=20)

    submit_btn.click(
        fn=transcribe_audio, 
        inputs=[record, filein], 
        outputs=output
    )

demo.launch()