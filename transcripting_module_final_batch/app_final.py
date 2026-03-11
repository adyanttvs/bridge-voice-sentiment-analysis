import os
import shutil
import time
import json
import base64
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from dotenv import load_dotenv
load_dotenv()

import dash
from dash import dcc, html, Input, Output, State, no_update, dash_table, callback_context, DiskcacheManager
import diskcache
from groq import Groq

# Configure Background Callbacks
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("[CRITICAL] GROQ_API_KEY not found in environment variables.")

client = Groq(api_key=GROQ_API_KEY)

# Initialize Dash App
app = dash.Dash(__name__, external_scripts=["https://cdn.tailwindcss.com"], suppress_callback_exceptions=True, background_callback_manager=background_callback_manager)

# Allow duplicate outputs for our newly injected callbacks on the same UI components
app.config.prevent_initial_callbacks = 'initial_duplicate'
app.title = "Bridge | Voice Sentiment Analysis (Dash)"

# Layout construction for Single Analysis Tab
single_analysis_layout = html.Div(className="grid grid-cols-1 md:grid-cols-12 gap-8 mt-6", children=[
    # LEFT COLUMN
    html.Div(className="col-span-1 md:col-span-3 flex flex-col gap-6", children=[
        # Input Panel
        html.Div(className="bg-white rounded-xl shadow-sm border border-slate-200 p-6", children=[
            html.H3("Audio Input", className="text-xs font-bold uppercase tracking-wider text-gray-500 mb-4"),
            dcc.Upload(
                id='upload-audio',
                accept='audio/*',
                children=html.Div([
                    html.Div("📤", className="text-3xl mb-2 text-gray-400"),
                    html.P("Click or drag & drop audio file", className="text-xs text-gray-500"),
                    html.P("MP3, WAV, M4A", className="text-[10px] text-gray-400 mt-1")
                ], className="border-2 border-dashed border-slate-300 rounded-lg p-5 text-center cursor-pointer hover:border-green-500 hover:bg-green-50 transition-all"),
                multiple=False
            ),
            html.Div(id='file-status', className="mt-4 text-[10px] text-green-600 text-center truncate min-h-[14px]"),
            
            html.Div(className="mt-4 mb-4", children=[
                html.Label("Detection Mode", className="text-[10px] uppercase font-bold tracking-widest text-gray-500 mb-2 block"),
                dcc.Dropdown(
                    id='lang-select',
                    options=[
                        {'label': '✨ UNIVERSAL (AUTO → ENGLISH)', 'value': 'auto'},
                        {'label': 'KANNADA', 'value': 'kn'},
                        {'label': 'TAMIL', 'value': 'ta'},
                        {'label': 'TELUGU', 'value': 'te'},
                        {'label': 'HINDI', 'value': 'hi'},
                        {'label': 'MALAYALAM', 'value': 'ml'},
                        {'label': 'ENGLISH', 'value': 'en'}
                    ],
                    value='ta',  # User requested regional default
                    clearable=False,
                    className="text-xs text-gray-700"
                )
            ]),
            
            html.Button("⚡ START ANALYSIS", id="start-btn", className="w-full bg-gradient-to-r from-green-500 to-teal-500 py-3 rounded-lg font-bold text-white shadow-lg transition-all hover:scale-[1.02] text-sm")
        ]),

        # Quality Score
        html.Div(className="bg-green-50 rounded-xl shadow-sm border border-green-200 p-6", children=[
            html.H3("Agent Quality Score", className="text-xs font-bold uppercase tracking-wider text-green-700 mb-4"),
            html.Div(className="flex items-end gap-2 mb-6", children=[
                html.Span("--", id="quality-score", className="text-5xl font-bold text-gray-800"),
                html.Span("/ 10", className="text-xl text-gray-500 mb-1")
            ]),
            html.Div(id="criteria-list", className="space-y-5", children=html.Div("Awaiting data...", className="text-gray-500 italic text-xs"))
        ]),

        # Accuracy
        html.Div(className="bg-blue-50 rounded-xl shadow-sm border border-blue-200 p-6", children=[
            html.H3("Transcription Accuracy", className="text-xs font-bold uppercase tracking-wider text-blue-700 mb-4"),
            html.Div(className="flex items-end gap-2 mb-4", children=[
                html.Span("--", id="accuracy-score", className="text-4xl font-bold text-gray-800"),
                html.Span("%", className="text-lg text-gray-500 mb-1")
            ]),
            html.Div("AI Confidence Level", className="text-[10px] text-gray-500 uppercase tracking-widest")
        ]),

        # Risk & Compliance
        html.Div(className="bg-red-50 rounded-xl shadow-sm border border-red-200 p-6", children=[
            html.H3("🛡️ Risk & Compliance", className="text-xs font-bold uppercase tracking-wider text-red-700 mb-4 flex items-center gap-2"),
            html.Div(id="risk-flags", className="space-y-2", children=html.Div("No risks detected...", className="text-[10px] text-gray-500 italic"))
        ]),
    ]),

    # RIGHT COLUMN
    html.Div(className="col-span-1 md:col-span-9 flex flex-col gap-6 min-h-[500px]", children=[
        html.Div(className="grid grid-cols-1 lg:grid-cols-2 gap-6", children=[
            # Tone Chart
            html.Div(className="bg-indigo-50 rounded-xl shadow-sm border border-indigo-200 p-6", children=[
                html.Div(className="flex items-center justify-between mb-4", children=[
                    html.H3([
                        html.Span(className="inline-block w-2 h-2 rounded-full bg-indigo-500 animate-pulse mr-2"),
                        "Voice Tone Analysis"
                    ], className="text-xs font-bold uppercase tracking-wider text-indigo-700 flex items-center"),
                    html.Span("Awaiting...", id="dominant-emotion", className="text-[10px] font-bold uppercase tracking-widest text-indigo-600 bg-indigo-100 px-3 py-1 rounded-full")
                ]),
                html.Div(id='tone-chart', className="space-y-2 mt-2")
            ]),

            # Sentiment Journey
            html.Div(className="bg-emerald-50 rounded-xl shadow-sm border border-emerald-200 p-6", children=[
                html.H3("📈 Sentiment Journey Arc", className="text-xs font-bold uppercase tracking-wider text-emerald-700 mb-4 flex items-center gap-2"),
                html.Div(id='journey-chart', className="mt-2")
            ])
        ]),

        # Interaction Analysis
        html.Div(className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 flex flex-col relative overflow-hidden", style={"minHeight": "380px"}, children=[
            html.H2("Interaction Analysis", className="text-lg font-semibold text-gray-800 mb-4"),
            html.Div(id="audio-player-container", className="mb-4 hidden", children=html.Audio(id="audio-player", controls=True, className="w-full")),
            html.Div(id="engine-indicator-container", className="mb-2 hidden"),
            
            dcc.Loading(
                id="loading-shield",
                type="circle",
                color="#22c55e",
                children=html.Div(id="results-content", className="flex-1 overflow-y-auto text-sm leading-relaxed p-6 bg-slate-50 rounded-lg font-light whitespace-pre-wrap text-slate-700 relative", style={"minHeight": "280px"}, children=[
                    html.Div("Upload audio and hit Start Analysis...", className="flex items-center justify-center h-full text-gray-400 italic")
                ])
            )
        ])
    ])
])

batch_dashboard_layout = html.Div(className="mt-6 flex flex-col gap-6", children=[
    html.Div(className="bg-white rounded-xl shadow-sm border border-slate-200 p-6", children=[
        html.Div(className="flex justify-between items-center mb-6", children=[
            html.H2("Batch Analysis Dashboard", className="text-xl font-bold text-gray-800"),
            html.Div(className="flex gap-2", children=[
                html.Button("▶ Run Batch Processor", id="run-batch-btn", className="text-xs bg-green-500 text-white font-bold px-4 py-2 rounded shadow hover:bg-green-600 transition-all"),
                html.Button("Refresh Data", id="refresh-batch-btn", className="text-xs bg-blue-50 text-blue-600 px-4 py-2 rounded border border-blue-200 hover:bg-blue-100")
            ])
        ]),
        
        # Batch Excel Upload Section
        html.Div(className="mb-6 bg-slate-50 border border-slate-200 rounded-lg p-5", children=[
            html.H3("1. Upload Call Data (Excel/CSV)", className="text-sm font-bold text-slate-700 mb-3"),
            dcc.Upload(
                id='upload-batch-excel',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files', className="text-blue-600 cursor-pointer font-bold")
                ]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '2px', 'borderStyle': 'dashed',
                    'borderRadius': '8px', 'textAlign': 'center', 'margin': '10px 0',
                    'backgroundColor': '#ffffff', 'borderColor': '#cbd5e1',
                    'fontSize': '14px', 'color': '#64748b'
                },
                multiple=False
            ),
            html.Div(id='batch-upload-status', className="text-xs text-green-600 font-semibold mt-2 min-h-[16px]")
        ]),
        
        html.Div(className="p-3 bg-amber-50 border border-amber-200 text-amber-800 rounded-lg text-xs mb-4 flex items-center gap-2", children=[
            html.Span("⚠️", className="text-lg"),
            html.Span("WARNING: The Groq API has a strict 1-hour audio processing limit per key. If hit, the system automatically shifts to Local Models. Processing will be slower. Leave the browser tab open while running.", className="font-semibold")
        ]),
        
        html.Div(id="batch-progress-status", className="mb-4 text-sm font-bold text-indigo-600"),
        
        dcc.Loading(
            id="loading-batch-table",
            type="dot",
            color="#3b82f6",
            children=html.Div(id="batch-table-container", children=html.Div("Upload data and run the processor to view results.", className="text-gray-500 italic"))
        )
    ]),
    
    html.Div(id="batch-details-panel", className="hidden", children=[
        html.Div(className="grid grid-cols-1 md:grid-cols-3 gap-6", children=[
            # Summary Score Card
            html.Div(className="col-span-1 bg-gradient-to-br from-green-50 to-emerald-100 rounded-xl shadow-sm border border-green-200 p-6 flex flex-col items-center justify-center", children=[
                html.H3("Total Score", className="text-xs font-bold uppercase tracking-wider text-green-700 mb-2"),
                html.Div(id="dash-score", className="text-6xl font-black text-gray-800 mb-2"),
                html.Div(id="dash-sentiment", className="px-3 py-1 bg-white rounded-full text-xs font-bold shadow-sm")
            ]),
            
            # Details Card
            html.Div(className="col-span-1 md:col-span-2 bg-white rounded-xl shadow-sm border border-slate-200 p-6", children=[
                html.H3("AI Summary & Insights", className="text-sm font-bold text-gray-800 mb-4 border-b pb-2"),
                html.P(id="dash-summary", className="text-slate-600 italic mb-4"),
                
                html.H4("Risk Flags", className="text-xs font-bold uppercase text-red-600 mb-2"),
                html.Div(id="dash-risks", className="text-sm text-gray-800")
            ])
        ]),
        
        # --- NEW MASSIVE EMBEDDED SINGLE ANALYSIS FOR BATCH ---
        html.Div(className="grid grid-cols-1 md:grid-cols-12 gap-6 mt-6", children=[
            # LEFT COLUMN (Scores & Risks)
            html.Div(className="col-span-1 md:col-span-4 flex flex-col gap-6", children=[
                # Quality Score
                html.Div(className="bg-green-50 rounded-xl shadow-sm border border-green-200 p-6", children=[
                    html.H3("Agent Quality Score", className="text-xs font-bold uppercase tracking-wider text-green-700 mb-4"),
                    html.Div(id="batch-criteria-list", className="space-y-5", children=html.Div("Awaiting data...", className="text-gray-500 italic text-xs"))
                ]),

                # Accuracy
                html.Div(className="bg-blue-50 rounded-xl shadow-sm border border-blue-200 p-6", children=[
                    html.H3("Transcription Accuracy", className="text-xs font-bold uppercase tracking-wider text-blue-700 mb-4"),
                    html.Div(className="flex items-end gap-2 mb-4", children=[
                        html.Span("--", id="batch-accuracy-score", className="text-4xl font-bold text-gray-800"),
                        html.Span("%", className="text-lg text-gray-500 mb-1")
                    ]),
                    html.Div("AI Confidence Level", className="text-[10px] text-gray-500 uppercase tracking-widest")
                ])
            ]),

            # RIGHT COLUMN (Charts & Transcripts)
            html.Div(className="col-span-1 md:col-span-8 flex flex-col gap-6", children=[
                html.Div(className="grid grid-cols-1 lg:grid-cols-2 gap-6", children=[
                    # Tone Chart
                    html.Div(className="bg-indigo-50 rounded-xl shadow-sm border border-indigo-200 p-6", children=[
                        html.Div(className="flex items-center justify-between mb-4", children=[
                            html.H3([
                                html.Span(className="inline-block w-2 h-2 rounded-full bg-indigo-500 animate-pulse mr-2"),
                                "Voice Tone Analysis"
                            ], className="text-xs font-bold uppercase tracking-wider text-indigo-700 flex items-center"),
                            html.Span("Awaiting...", id="batch-dominant-emotion", className="text-[10px] font-bold uppercase tracking-widest text-indigo-600 bg-indigo-100 px-3 py-1 rounded-full")
                        ]),
                        html.Div(id='batch-tone-chart', className="space-y-2 mt-2")
                    ]),

                    # Sentiment Journey
                    html.Div(className="bg-emerald-50 rounded-xl shadow-sm border border-emerald-200 p-6", children=[
                        html.H3("📈 Sentiment Journey Arc", className="text-xs font-bold uppercase tracking-wider text-emerald-700 mb-4 flex items-center gap-2"),
                        html.Div(id='batch-journey-chart', className="mt-2")
                    ])
                ]),
                
                # Interaction Analysis (Transcripts)
                html.Div(className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 flex flex-col relative overflow-hidden", children=[
                    html.H2("Complete Conversation Transcript", className="text-lg font-semibold text-gray-800 mb-4"),
                    html.Div(id="batch-audio-player-container", className="mb-4 hidden", children=html.Audio(id="batch-audio-player", controls=True, className="w-full")),
                    html.Div(id="batch-results-content", className="overflow-y-auto text-sm leading-relaxed p-6 bg-slate-50 rounded-lg font-light whitespace-pre-wrap text-slate-700 max-h-[500px]", children=[
                        html.Div("Select a row in the table to view the full dialogue...", className="flex items-center justify-center h-full text-gray-400 italic")
                    ])
                ])
            ])
        ])
    ])
])

app.layout = html.Div(className="p-4 md:p-8 bg-slate-50 min-h-screen text-slate-800 font-sans", children=[
    html.Div(className="max-w-7xl mx-auto", children=[
        # Header
        html.Div(className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8", children=[
            html.Div(className="flex items-center gap-3", children=[
                html.Div("📡", className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center text-xl text-white shadow-lg"),
                html.H1([
                    "Bridge ", 
                    html.Span("Voice", className="font-light text-gray-500"), 
                    " Sentiment Analysis ",
                    html.Span("Dash v5.0", className="text-xs bg-blue-100 text-blue-600 px-2 py-0.5 rounded ml-2")
                ], className="text-xl md:text-2xl font-bold text-gray-800")
            ]),
            html.Div([
                html.Div(className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"),
                "Bridge AI Cloud Intelligence"
            ], className="px-3 py-1 bg-blue-50 rounded-full text-[10px] md:text-xs flex items-center gap-2 text-blue-700")
        ]),

        dcc.Tabs(id="tabs", value='tab-batch', className="custom-tabs font-bold mb-4", children=[
            dcc.Tab(label='📊 Batch Dashboard', value='tab-batch', className="p-4 rounded-t-lg bg-gray-200 border-none text-gray-600", selected_className="p-4 rounded-t-lg bg-white border-t-4 border-t-blue-500 text-blue-700 shadow-sm"),
            dcc.Tab(label='🎙️ Single Analysis', value='tab-single', className="p-4 rounded-t-lg bg-gray-200 border-none text-gray-600", selected_className="p-4 rounded-t-lg bg-white border-t-4 border-t-blue-500 text-blue-700 shadow-sm")
        ]),
        
        html.Div(id='tabs-content')
    ])
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-single':
        return single_analysis_layout
    elif tab == 'tab-batch':
        return batch_dashboard_layout

def process_audio_local(temp_filename: str, lang: str):
    """Fallback logic using entirely local models."""
    print("[FALLBACK] Running local Whisper for transcription...")
    import whisper
    
    # Load whisper locally (small model to balance speed and quality)
    local_whisper = whisper.load_model("base")
    # For local Whisper, '' auto-detects, but we can set language if provided
    options = {}
    if lang != "auto":
        options["language"] = lang
        
    result = local_whisper.transcribe(temp_filename, **options)
    transcribed_text = result["text"]
    detected_lang_code = result.get("language", "en")
    
    print(f"[FALLBACK] Transcribed locally: {len(transcribed_text)} characters.")
    
    print("[FALLBACK] Running local pipeline for auditing...")
    # Load lightweight local text model
    from transformers import pipeline
    pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM-135M", device_map="auto")
    
    prompt = f"""Evaluate this call transcript and output JSON ONLY.
Transcript: {transcribed_text}
Output must perfectly match this structure:
{{"total_score": 8, "criteria_breakdown": {{"brand_greeting": 2, "solution_clarity": 2, "professional_tone": 2, "compliance": 2, "quality_closure": 0}}, "summary": "Short agent summary.", "sentiment": {{"label": "neu"}}, "risk_flags": [], "sentiment_journey": [0,0.5,0.7], "transcription_confidence": 90, "emotion_scores": {{"angry": 0.1, "calm": 0.8}}}}
"""
    
    out = pipe(prompt, max_new_tokens=256, return_full_text=False)
    generated_text = out[0]['generated_text']
    
    # Attempt to safely parse the JSON output from the local LLM
    import re
    json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
    
    if json_match:
        try:
            audit_data = json.loads(json_match.group(0))
        except Exception:
            audit_data = {"total_score": 5, "summary": "Local parsing failed.", "sentiment": {"label": "neu"}}
    else:
        audit_data = {"total_score": 5, "summary": "Local generation failed to produce JSON.", "sentiment": {"label": "neu"}}
        
    diarized_segments = []
    return {
        "text": transcribed_text,
        "native_text": transcribed_text,
        "diarized_segments": diarized_segments,
        "audit": audit_data,
        "detected_language_code": detected_lang_code,
    }

def process_audio_file(temp_filename: str, lang: str):
    """Core logic extracted from app_v3_main.py"""
    multilingual_prompt = (
        "Contact Center customer service call - Agent side only. "
        "The speaker is a customer care agent using Tamil, Telugu, Hindi, Malayalam, Kannada, or English. "
        "Common words: hello, hi, yes, no, ok, okay, sure, thanks, thank you, "
        "The agent is assisting a customer with professional services."
    )

    trans_params_base = {
        "model": "whisper-large-v3",
        "prompt": multilingual_prompt,
        "temperature": 0,
        "response_format": "verbose_json",
    }

    native_text = None
    english_text = None

    with open(temp_filename, "rb") as af1:
        params1 = dict(trans_params_base, file=af1)
        if lang != "auto":
            params1["language"] = lang
            
        # Always transcribe first to properly detect the native language
        native_response = client.audio.transcriptions.create(**params1)
        native_text = native_response.text
        
        # In verbose_json, Groq correctly identifies the language object
        detected_lang_code = getattr(native_response, "language", None)
        if not detected_lang_code:
            detected_lang_code = lang if lang != "auto" else "English (Fallback)"
            
        transcript_response = native_response

    # If the detected language is not English, generate the translation
    english_text = None
    if "en" not in detected_lang_code.lower() and "english" not in detected_lang_code.lower():
        try:
            with open(temp_filename, "rb") as af2:
                params2 = dict(trans_params_base, file=af2)
                eng_response = client.audio.translations.create(**params2)
                english_text = eng_response.text
        except Exception as e:
            print(f"Translation API error: {e}")

    transcribed_text = english_text if english_text else native_text
    
    # Audit logic
    system_prompt = """
    You are an Expert Quality & Compliance Auditor for GoBumpr Customer Care (formerly MyTVS).
    Your goal is to evaluate the performance of the Customer Care Service Agent ONLY based on their spoken side of the call.
    STRICT RULES:
    1. YOU ARE REVIEWING THE AGENT'S AUDIO ONLY. 
    2. DO NOT comment on the customer.
    CRITICAL COMPLIANCE STEP: Identify "Agent Risk Flags" (mentions competitors, rude, incorrect info).
    SCORING RULES (10 pts total, each criteria is 0 or 2 pts, NO partial credit):
    - brand_greeting -> 2 pts
    - solution_clarity -> 2 pts
    - professional_tone -> 2 pts
    - compliance -> 2 pts
    - quality_closure -> 2 pts
       Give 2 if: agent said "thank you" or "thanks".
       Give 0 ONLY if: agent did not say thank you.
    Return JSON ONLY:
    {
        "total_score": NUMBER,
        "criteria_breakdown": {"brand_greeting": 2, "solution_clarity": 2, "professional_tone": 2, "compliance": 2, "quality_closure": 2},
        "summary": "Focus ONLY on the AGENT. 2 Sentences max.",
        "sentiment": {"label": "pos/neu/neg", "score_pos": 0.0, "score_neu": 0.0, "score_neg": 0.0},
        "risk_flags": ["LIST", "OF", "AGENT", "RED", "FLAGS", "OR", "EMPTY"],
        "sentiment_journey": [0, 0.2, 0.5, 0.3, 0.4, 0.8],
        "transcription_confidence": NUMBER (0-100),
        "emotion_scores": {"angry": 0.1, "calm": 0.1, "disgust": 0.1, "fearful": 0.1, "happy": 0.1, "neutral": 0.1, "sad": 0.1, "surprised": 0.1}
    }
    """
    audit_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Transcript:\n\n{transcribed_text}"}
        ],
        response_format={ "type": "json_object" }
    )
    
    audit_data = json.loads(audit_response.choices[0].message.content)

    # TOKEN OPTIMIZATION: Removed secondary Llama call for Diarization.
    # We will simply assign segments alternatingly to save 50% of the API tokens.
    diarized_segments = []
    raw_segments = getattr(transcript_response, "segments", [])
    if raw_segments:
        for i, s in enumerate(raw_segments):
            # Alternating heuristic to approximate turns without using LLM
            speaker = "AGENT" if i % 2 == 0 else "CUSTOMER" 
            diarized_segments.append({
                "id": i,
                "speaker": speaker,
                "text": s.get('text', '').strip()
            })

    return {
        "text": transcribed_text,
        "native_text": native_text,
        "diarized_segments": diarized_segments,
        "audit": audit_data,
        "detected_language_code": detected_lang_code,
    }


# -------- CALLBACKS SEPARATED INTO RESPECTIVE TABS --------

@app.callback(
    Output('file-status', 'children'),
    Input('upload-audio', 'filename')
)
def update_file_status(filename):
    if filename:
        return f"📁 {filename}"
    return dash.no_update

@app.callback(
    Output('batch-upload-status', 'children'),
    Input('upload-batch-excel', 'filename')
)
def update_batch_upload_status(filename):
    if filename:
        return f"✅ Ready to process: {filename}"
    return "⚠️ Awaiting file upload..."

@app.callback(
    [
        Output('quality-score', 'children'),
        Output('criteria-list', 'children'),
        Output('accuracy-score', 'children'),
        Output('risk-flags', 'children'),
        Output('tone-chart', 'figure'),
        Output('dominant-emotion', 'children'),
        Output('journey-chart', 'figure'),
        Output('results-content', 'children'),
        Output('audio-player-container', 'className'),
        Output('audio-player', 'src'),
        Output('engine-indicator-container', 'children'),
        Output('engine-indicator-container', 'className')
    ],
    Input('start-btn', 'n_clicks'),
    State('upload-audio', 'contents'),
    State('upload-audio', 'filename'),
    State('lang-select', 'value'),
    prevent_initial_call=True
)
def run_analysis(n_clicks, contents, filename, lang):
    if not contents:
        return ["--", html.Div("Please upload audio first.", className="italic"), "--", html.Div("No risks detected...", className="italic text-gray-400")] + [dash.no_update]*6 + [dash.no_update, "hidden"]

    # Decode and save audio safely
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    temp_filename = f"dash_temp_{int(time.time())}.mp3"
    
    try:
        with open(temp_filename, "wb") as f:
            f.write(decoded)
            
        success = False
        engine_used = "groq"
        
        # Try Cloud API first
        try:
            data = process_audio_file(temp_filename, lang)
            success = True
        except Exception as api_err:
            err_msg = str(api_err)
            if '429' in err_msg or 'rate_limit_exceeded' in err_msg or 'timeout' in err_msg.lower() or 'read' in err_msg.lower():
                print("[SINGLE] API Error! Triggering LOCAL FALLBACK model...")
                engine_used = "local"
                data = process_audio_local(temp_filename, lang)
                success = True
            else:
                raise api_err
                
        audit = data['audit']
        
        # UI Builders
        # Quality Score
        quality_score = str(audit.get('total_score', '--'))
        
        # Criteria List
        criteria = audit.get('criteria_breakdown', {})
        criteria_list = []
        for k, v in criteria.items():
            name = k.replace('_', ' ').title()
            color = "bg-green-500" if v > 1 else "bg-red-500"
            text_color = "text-green-600" if v > 1 else "text-red-600"
            criteria_list.append(html.Div([
                html.Div([
                    html.Span(name, className="capitalize text-gray-600 text-[10px] uppercase font-black tracking-widest"),
                    html.Span(f"{v}/2", className=f"{text_color} font-bold")
                ], className="flex justify-between items-center mb-1"),
                html.Div(className="h-1 bg-gray-200 rounded-full overflow-hidden", children=[
                    html.Div(className=f"h-full {color} transition-all", style={"width": f"{v*50}%"})
                ])
            ]))
            
        # Accuracy
        acc = str(audit.get('transcription_confidence', '--'))
        
        # Risks
        flags = audit.get('risk_flags', [])
        if not flags:
            risk_ui = html.Div("✅ ALL COMPLIANT", className="text-[10px] text-green-600 font-bold uppercase tracking-widest")
        else:
            risk_ui = []
            for f in flags:
                risk_ui.append(html.Div([html.Span("🚩", className="animate-pulse text-red-500 mr-2"), f], 
                    className="flex items-center bg-red-100 border border-red-200 p-2 rounded-lg text-[10px] text-red-700 font-bold uppercase tracking-wider mb-2"))
                    
        # Tone Bars (HTML CSS)
        emotions = audit.get('emotion_scores', {})
        emotion_colors = {'angry': '#ef4444', 'calm': '#6366f1', 'disgust': '#a855f7', 'fearful': '#fb923c', 'happy': '#22c55e', 'neutral': '#64748b', 'sad': '#3b82f6', 'surprised': '#fbbf24'}
        fig_tone = []
        for emo, val in emotions.items():
            pct = int(val * 100)
            bar_color = emotion_colors.get(emo, '#94a3b8')
            fig_tone.append(html.Div(className="flex items-center gap-2", children=[
                html.Span(emo.title(), className="text-[10px] text-gray-600 w-16 uppercase font-bold shrink-0"),
                html.Div(className="flex-1 bg-gray-200 rounded-full h-2 overflow-hidden", children=[
                    html.Div(style={"width": f"{pct}%", "backgroundColor": bar_color}, className="h-full rounded-full")
                ]),
                html.Span(f"{pct}%", className="text-[10px] text-gray-500 w-7 text-right")
            ]))
        
        dom_emotion = max(emotions, key=emotions.get).upper() if emotions else "UNKNOWN"
        
        # Journey Sparkline (HTML dots)
        journey = audit.get('sentiment_journey', [])
        def sentiment_dot(val):
            if val > 0.3: return "bg-green-500"
            elif val < -0.3: return "bg-red-500"
            return "bg-yellow-400"
        
        fig_journey = html.Div(className="flex items-end gap-1 h-16", children=[
            html.Div(className=f"flex-1 rounded-sm {sentiment_dot(v)}",
                style={"height": f"{int((v + 1) / 2 * 100)}%", "minHeight": "4px"})
            for v in journey
        ])
        
        # Transcript Content
        transcript_ui = []
        if data['diarized_segments']:
            bubbles = []
            for seg in data['diarized_segments']:
                is_agent = str(seg.get('speaker', '')).upper() == 'AGENT'
                bg = 'bg-blue-50 border-blue-200 text-blue-900 rounded-br-sm' if is_agent else 'bg-slate-50 border-slate-200 text-slate-800 rounded-bl-sm'
                align = 'items-end' if is_agent else 'items-start'
                flex_dir = 'flex-row-reverse' if is_agent else 'flex-row'
                avatar = 'AG' if is_agent else 'CX'
                avatar_bg = 'bg-blue-100 text-blue-700' if is_agent else 'bg-slate-200 text-slate-700'
                
                bubbles.append(html.Div(className=f"flex gap-3 mb-4 {flex_dir}", children=[
                    html.Div(avatar, className=f"w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center text-[10px] font-bold {avatar_bg}"),
                    html.Div(className=f"flex flex-col {align}", children=[
                        html.Span("Agent" if is_agent else "Customer", className="text-[9px] font-bold uppercase text-slate-400 mb-1 tracking-wider"),
                        html.Div(seg.get('text', ''), className=f"p-3 rounded-2xl border text-sm max-w-[85%] {bg}")
                    ])
                ]))
            
            transcript_ui.append(html.Div(className="mb-8", children=[
                html.H3("💬 SPEAKER BREAKDOWN", className="text-indigo-700 font-bold uppercase text-[10px] tracking-[0.2em] mb-4"),
                html.Div(bubbles, className="p-4 bg-white border border-slate-200 rounded-xl max-h-[320px] overflow-y-auto")
            ]))
            
        transcript_ui.append(html.Div(className="p-5 bg-green-50 border border-green-200 rounded-xl mb-6", children=[
            html.H3("🇬🇧 FULL TRANSCRIPT", className="text-green-700 font-bold uppercase text-[10px] tracking-[0.2em] mb-3"),
            html.P(data['text'], className="text-slate-800 leading-relaxed font-light")
        ]))
        
        transcript_ui.append(html.Div(className="p-8 bg-blue-50 rounded-[16px] border border-blue-200 shadow-md", children=[
            html.H3("🛡️ AI AUDITOR INSIGHT", className="text-blue-700 font-bold mb-4 flex items-center gap-3"),
            html.P(f"\"{audit.get('summary', '')}\"", className="text-slate-700 text-lg italic leading-relaxed font-light mb-6")
        ]))
        
        # Engine UI Builder
        if engine_used == "groq":
            engine_ui = html.Div(className="bg-purple-100 border border-purple-200 text-purple-700 px-3 py-2 rounded-lg text-xs font-bold flex items-center gap-2", children=[
                html.Span("⚡", className="animate-pulse"),
                "Processed via Groq Cloud API (Ultra-Fast)"
            ])
        else:
            engine_ui = html.Div(className="bg-amber-100 border border-amber-200 text-amber-700 px-3 py-2 rounded-lg text-xs font-bold flex items-center gap-2", children=[
                html.Span("🖥️", className="animate-pulse"),
                "Processed via Local Fallback Engine (Offline Modes Active)"
            ])
        
        return quality_score, criteria_list, acc, risk_ui, fig_tone, dom_emotion, fig_journey, transcript_ui, "mb-4 block", contents, engine_ui, "mb-2 block flex justify-end"
        
    except Exception as e:
        print(e)
        return ["Error", html.Div(str(e)), "--", html.Div("Error evaluating", className="italic text-red-500")] + [dash.no_update]*6 + [dash.no_update, "hidden"]
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.callback(
    Output('batch-progress-status', 'children'),
    Input('run-batch-btn', 'n_clicks'),
    State('upload-batch-excel', 'contents'),
    State('upload-batch-excel', 'filename'),
    prevent_initial_call=True
)
def run_batch_process(n_clicks, excel_contents, excel_filename):
    """Integrates the external batch processor directly into the Dash runtime using an uploaded file."""
    if not n_clicks:
        return dash.no_update
        
    if not excel_contents:
        return "⚠️ Error: Please upload an Excel or CSV file first!"
        
    print("[BATCH] Batch Processing triggered from UI...")
    import httpx
    import re
    import io
    
    # Dynamically isolate batch results by uploaded filename
    base_name = os.path.splitext(excel_filename)[0]
    sanitized_name = "".join([c for c in base_name if c.isalpha() or c.isdigit() or c=='_']).rstrip()
    if not sanitized_name:
        sanitized_name = "batch"
        
    batch_dir = "./batch_data"
    os.makedirs(batch_dir, exist_ok=True)
    output_path = f"{batch_dir}/{sanitized_name}_results.csv"
    
    # Parse the uploaded file
    try:
        content_type, content_string = excel_contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'csv' in excel_filename.lower():
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in excel_filename.lower():
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return "⚠️ Error: Unsupported file format. Please upload .xlsx or .csv"
    except Exception as parse_e:
        return f"⚠️ Error reading uploaded file: {parse_e}"
        
    if 'Length In Sec' not in df.columns or 'Phone Number' not in df.columns or 'Location' not in df.columns:
        return "⚠️ Error: Uploaded file is missing required columns ('Phone Number', 'Length In Sec', 'Location')."
        
    filtered_df = df[df['Length In Sec'] > 100].copy()
    total_files = len(filtered_df)
    
    # Load past results to avoid reprocessing
    existing_phones = set()
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            existing_phones = set(existing_df['Phone Number'].astype(str))
            # Put previous data in results list so it's not overwritten
            results = existing_df.to_dict('records')
        except:
            results = []
    else:
        results = []
        
    total_processed_seconds = 0
    max_seconds = 3500  # Leave buffer for 1-hour Groq API limit
    processed_count = 0
    
    for idx, row in filtered_df.iterrows():
        phone = str(row.get('Phone Number', 'Unknown'))
        
        # Skip if already in CSV
        if phone in existing_phones:
            continue
            
        url = row.get('Location', '')
        length = row.get('Length In Sec', 0)
        
        if pd.isna(url) or not str(url).startswith('http'):
            continue
            
        if total_processed_seconds + length > max_seconds:
             msg = f"⚠️ [Limit Reached]: Processed {processed_count} files before hitting safety limit. Done."
             print(msg)
             break
             
        print(f"\n[BATCH] Processing {phone} (Len: {length}s)")
        temp_filename = f"dash_batch_{int(time.time())}_{idx}.mp3"
        
        try:
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
                with open(temp_filename, "wb") as f:
                    f.write(resp.content)
            
            # API processing loop
            success = False
            for attempt in range(2):
                try:
                    data = process_audio_file(temp_filename, "auto")
                    success = True
                    break
                except Exception as api_err:
                    err_msg = str(api_err)
                    # Use fallback for rate limit
                    if '429' in err_msg or 'rate_limit_exceeded' in err_msg:
                        print("[BATCH] API Rate Limit hit! Triggering LOCAL FALLBACK model...")
                        try:
                            data = process_audio_local(temp_filename, "auto")
                            success = True
                            print("[BATCH] Local model completed successfully.")
                            break
                        except Exception as local_err:
                            print(f"[BATCH] Local fallback failed too: {local_err}")
                            break
                    elif 'timeout' in err_msg.lower() or 'read' in err_msg.lower():
                        print(f"[BATCH] Timeout on attempt {attempt+1}. Retrying...")
                        time.sleep(5)
                        if attempt == 1:
                            print("[BATCH] Trying local model due to persistent API timeouts...")
                            data = process_audio_local(temp_filename, "auto")
                            success = True
                    else:
                        break # Other errors skip retry
            
            if success:
                audit = data.get('audit', {})
                
                # Save full JSON to local cache for Single Analysis tab integration
                try:
                    with open(f"./batch_data/{phone}.json", "w", encoding="utf-8") as jf:
                        json.dump(data, jf)
                except Exception as e:
                    print(f"[BATCH] Failed to cache JSON for {phone}: {e}")
                
                results.append({
                    'Phone Number': phone,
                    'Length In Sec': length,
                    'Audio URL': url,
                    'Total Score': audit.get('total_score', 'N/A'),
                    'Risk Flags': ', '.join(audit.get('risk_flags', [])),
                    'Sentiment Label': audit.get('sentiment', {}).get('label', 'Unknown'),
                    'Summary': audit.get('summary', ''),
                    'Detected Language': data.get('detected_language_code', 'Unknown'),
                    'Processed By': 'Local Fallback' if '429' in str(locals().get('err_msg','')) else 'Groq API'
                })
                total_processed_seconds += length
                processed_count += 1
                print(f" -> Success Score: {audit.get('total_score')}")
                
                # Continuously save to CSV to prevent data loss
                pd.DataFrame(results).to_csv(output_path, index=False)
                
        except Exception as e:
            print(f"[BATCH] Critical Error on {phone}: {e}")
            
        finally:
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except:
                    pass
    
    return html.Div([
        html.Span("✅ Batch completed! ", className="text-green-600"),
        html.Span(f"Processed {processed_count} new files total.")
    ])


@app.callback(
    Output('batch-table-container', 'children'),
    Input('refresh-batch-btn', 'n_clicks'),
    State('upload-batch-excel', 'filename')
)
def load_batch_table(n_clicks, filename):
    if not filename:
        return html.Div("Upload an Excel file to see its batch results.", className="text-gray-500 italic")
        
    base_name = os.path.splitext(filename)[0]
    sanitized_name = "".join([c for c in base_name if c.isalpha() or c.isdigit() or c=='_']).rstrip()
    if not sanitized_name:
        sanitized_name = "batch"
        
    batch_dir = "./batch_data"
    os.makedirs(batch_dir, exist_ok=True)
    csv_path = f"{batch_dir}/{sanitized_name}_results.csv"
    
    if not os.path.exists(csv_path):
        return html.Div(f"No batch tracking file found for {filename}. Run batch analyzer first.", className="text-red-500")
        
    df = pd.read_csv(csv_path)
    
    return html.Div(className="mt-2", children=[
        html.P("💡 Click on any row below to reveal the AI Summary, Risk Flags, and an Audio Player for that specific call.", className="text-xs text-blue-600 mb-3 font-semibold"),
        dash_table.DataTable(
        id='batch-table',
        columns=[
            {"name": i, "id": i} for i in ['Phone Number', 'Total Score', 'Sentiment Label', 'Detected Language', 'Processed By', 'Length In Sec']
        ],
        data=df.to_dict('records'),
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_header={
            'backgroundColor': '#f8fafc',
            'fontWeight': 'bold',
            'color': '#475569'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f1f5f9'
            }
        ],
        row_selectable="single",
        page_size=10
    )])


@app.callback(
    [
        Output('batch-details-panel', 'className'),
        Output('batch-criteria-list', 'children'),
        Output('batch-accuracy-score', 'children'),
        Output('batch-tone-chart', 'figure'),
        Output('batch-dominant-emotion', 'children'),
        Output('batch-journey-chart', 'figure'),
        Output('batch-results-content', 'children'),
        Output('batch-audio-player-container', 'className'),
        Output('batch-audio-player', 'src'),
        Output('dash-score', 'children'),
        Output('dash-sentiment', 'children'),
        Output('dash-summary', 'children'),
        Output('dash-risks', 'children')
    ],
    Input('batch-table', 'selected_rows'),
    State('batch-table', 'data'),
    prevent_initial_call=True
)
def update_batch_details(selected_rows, data):
    if not selected_rows or not data:
        return "hidden", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, "hidden", "", dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
    row = data[selected_rows[0]]
    phone = row.get('Phone Number', '')
    url = row.get('Audio URL', '')
    
    json_path = f"./batch_data/{phone}.json"
    
    if not os.path.exists(json_path):
        err_msg = html.Div(f"⚠️ Cached data ({phone}.json) not found. You must re-run the batch processor.", className="font-bold text-red-500 p-4")
        return "block mt-6", html.Div("Missing Data", className="italic text-red-500"), "--", dash.no_update, dash.no_update, dash.no_update, [err_msg], "hidden", "", "--", "--", "Missing Summary", "Missing Risks"
        
    try:
        with open(json_path, "r", encoding="utf-8") as jf:
            audit_data = json.load(jf)
            
        audit = audit_data.get('audit', {})
        
        # 1. Criteria List (Left Column)
        criteria = audit.get('criteria_breakdown', {})
        criteria_list = []
        for k, v in criteria.items():
            name = k.replace('_', ' ').title()
            color = "bg-green-500" if v > 1 else "bg-red-500"
            text_color = "text-green-600" if v > 1 else "text-red-600"
            criteria_list.append(html.Div([
                html.Div([
                    html.Span(name, className="capitalize text-gray-600 text-[10px] uppercase font-black tracking-widest"),
                    html.Span(f"{v}/2", className=f"{text_color} font-bold")
                ], className="flex justify-between items-center mb-1"),
                html.Div(className="h-1 bg-gray-200 rounded-full overflow-hidden", children=[
                    html.Div(className=f"h-full {color} transition-all", style={"width": f"{v*50}%"})
                ])
            ]))
            
        # 2. Accuracy
        acc = str(audit.get('transcription_confidence', '--'))
        
        # 3. Tone Bars (HTML CSS)
        emotions = audit.get('emotion_scores', {})
        emotion_colors = {'angry': '#ef4444', 'calm': '#6366f1', 'disgust': '#a855f7', 'fearful': '#fb923c', 'happy': '#22c55e', 'neutral': '#64748b', 'sad': '#3b82f6', 'surprised': '#fbbf24'}
        fig_tone = []
        for emo, val in emotions.items():
            pct = int(val * 100)
            bar_color = emotion_colors.get(emo, '#94a3b8')
            fig_tone.append(html.Div(className="flex items-center gap-2", children=[
                html.Span(emo.title(), className="text-[10px] text-gray-600 w-16 uppercase font-bold shrink-0"),
                html.Div(className="flex-1 bg-gray-200 rounded-full h-2 overflow-hidden", children=[
                    html.Div(style={"width": f"{pct}%", "backgroundColor": bar_color}, className="h-full rounded-full")
                ]),
                html.Span(f"{pct}%", className="text-[10px] text-gray-500 w-7 text-right")
            ]))

        dom_emotion = max(emotions, key=emotions.get).upper() if emotions else "UNKNOWN"

        # 4. Journey Sparkline (HTML dots)
        journey = audit.get('sentiment_journey', [])
        def sentiment_dot(val):
            if val > 0.3: return "bg-green-500"
            elif val < -0.3: return "bg-red-500"
            return "bg-yellow-400"

        fig_journey = html.Div(className="flex items-end gap-1 h-16", children=[
            html.Div(className=f"flex-1 rounded-sm {sentiment_dot(v)}",
                style={"height": f"{int((v + 1) / 2 * 100)}%", "minHeight": "4px"})
            for v in journey
        ])
        
        # 5. Transcript Content
        transcript_ui = []
        if audit_data.get('diarized_segments'):
            bubbles = []
            for seg in audit_data['diarized_segments']:
                is_agent = str(seg.get('speaker', '')).upper() == 'AGENT'
                bg = 'bg-blue-50 border-blue-200 text-blue-900 rounded-br-sm' if is_agent else 'bg-slate-50 border-slate-200 text-slate-800 rounded-bl-sm'
                align = 'items-end' if is_agent else 'items-start'
                flex_dir = 'flex-row-reverse' if is_agent else 'flex-row'
                avatar = 'AG' if is_agent else 'CX'
                avatar_bg = 'bg-blue-100 text-blue-700' if is_agent else 'bg-slate-200 text-slate-700'
                
                bubbles.append(html.Div(className=f"flex gap-3 mb-4 {flex_dir}", children=[
                    html.Div(avatar, className=f"w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center text-[10px] font-bold {avatar_bg}"),
                    html.Div(className=f"flex flex-col {align}", children=[
                        html.Span("Agent" if is_agent else "Customer", className="text-[9px] font-bold uppercase text-slate-400 mb-1 tracking-wider"),
                        html.Div(seg.get('text', ''), className=f"p-3 rounded-2xl border text-sm max-w-[85%] {bg}")
                    ])
                ]))
            
            transcript_ui.append(html.Div(className="mb-8", children=[
                html.H3("💬 SPEAKER BREAKDOWN", className="text-indigo-700 font-bold uppercase text-[10px] tracking-[0.2em] mb-4"),
                html.Div(bubbles, className="p-4 bg-white border border-slate-200 rounded-xl max-h-[350px] overflow-y-auto")
            ]))
            
        transcript_ui.append(html.Div(className="p-5 bg-green-50 border border-green-200 rounded-xl mb-6", children=[
            html.H3("🇬🇧 FULL TRANSCRIPT", className="text-green-700 font-bold uppercase text-[10px] tracking-[0.2em] mb-3"),
            html.P(audit_data.get('text', 'No transcript available.'), className="text-slate-800 leading-relaxed font-light")
        ]))
        
        transcript_ui.append(html.Div(className="p-8 bg-blue-50 rounded-[16px] border border-blue-200 shadow-md", children=[
            html.H3("🛡️ AI AUDITOR INSIGHT", className="text-blue-700 font-bold mb-4 flex items-center gap-3"),
            html.P(f"\"{audit.get('summary', '')}\"", className="text-slate-700 text-lg italic leading-relaxed font-light mb-6")
        ]))
        
        dash_score_ui = str(audit.get('total_score', '--'))
        dash_sentiment_ui = audit.get('sentiment', {}).get('label', 'Unknown').upper()
        dash_summary_ui = audit.get('summary', 'No summary provided.')
        
        flags = audit.get('risk_flags', [])
        if not flags:
            dash_risks_ui = html.Div("✅ NO RISKS", className="text-[10px] text-green-600 font-bold uppercase tracking-widest")
        else:
            dash_risks_ui = [html.Div(f"🚩 {f}", className="text-red-600 text-xs mb-1") for f in flags]

        return "block mt-6", criteria_list, acc, fig_tone, dom_emotion, fig_journey, transcript_ui, "mb-4 block", url, dash_score_ui, dash_sentiment_ui, dash_summary_ui, dash_risks_ui
        
    except Exception as e:
        err_msg = html.Div(f"Internal JSON read error: {e}", className="font-bold text-red-500 rounded p-4")
        return "block mt-6", html.Div("Missing JSON", className="italic"), "--", dash.no_update, dash.no_update, dash.no_update, [err_msg], "hidden", "", "--", "--", "Error", "Error"


if __name__ == '__main__':
    app.run(debug=True, port=8050)
