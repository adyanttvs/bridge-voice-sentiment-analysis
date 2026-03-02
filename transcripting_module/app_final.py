import os
import shutil
import time
import json
import base64
import plotly.graph_objects as go
import numpy as np

from dotenv import load_dotenv
load_dotenv()

import dash
from dash import dcc, html, Input, Output, State, no_update
from groq import Groq

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("[CRITICAL] GROQ_API_KEY not found in environment variables.")

client = Groq(api_key=GROQ_API_KEY)

# Initialize Dash App
app = dash.Dash(__name__, external_scripts=["https://cdn.tailwindcss.com"], suppress_callback_exceptions=True)
app.title = "GoBumpr | Voice Sentiment Analysis (Dash)"

# Layout
app.layout = html.Div(className="p-4 md:p-8 bg-slate-50 min-h-screen text-slate-800 font-sans", children=[
    html.Div(className="max-w-7xl mx-auto", children=[
        # Header
        html.Div(className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8", children=[
            html.Div(className="flex items-center gap-3", children=[
                html.Div("📡", className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center text-xl text-white shadow-lg"),
                html.H1([
                    "GoBumpr ", 
                    html.Span("Voice", className="font-light text-gray-500"), 
                    " Sentiment Analysis ",
                    html.Span("Dash v4.0", className="text-xs bg-blue-100 text-blue-600 px-2 py-0.5 rounded ml-2")
                ], className="text-xl md:text-2xl font-bold text-gray-800")
            ]),
            html.Div([
                html.Div(className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"),
                "GoBumpr AI Cloud Intelligence"
            ], className="px-3 py-1 bg-blue-50 rounded-full text-[10px] md:text-xs flex items-center gap-2 text-blue-700")
        ]),

        html.Div(className="grid grid-cols-1 md:grid-cols-12 gap-8", children=[
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
                            value='auto',
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
                        dcc.Graph(id='tone-chart', className="h-48", config={'displayModeBar': False})
                    ]),

                    # Sentiment Journey
                    html.Div(className="bg-emerald-50 rounded-xl shadow-sm border border-emerald-200 p-6", children=[
                        html.H3("📈 Sentiment Journey Arc", className="text-xs font-bold uppercase tracking-wider text-emerald-700 mb-4 flex items-center gap-2"),
                        dcc.Graph(id='journey-chart', className="h-48", config={'displayModeBar': False})
                    ])
                ]),

                # Interaction Analysis
                html.Div(className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 flex flex-col relative overflow-hidden", style={"minHeight": "380px"}, children=[
                    html.H2("Interaction Analysis", className="text-lg font-semibold text-gray-800 mb-4"),
                    html.Div(id="audio-player-container", className="mb-4 hidden", children=html.Audio(id="audio-player", controls=True, className="w-full")),
                    
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
    ])
])

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
            native_response = client.audio.transcriptions.create(**params1)
            native_text = native_response.text
            detected_lang_code = lang
            transcript_response = native_response
        else:
            english_response = client.audio.translations.create(**params1)
            english_text = english_response.text
            detected_lang_code = getattr(english_response, "language", "en")
            transcript_response = english_response

    if native_text and not english_text:
        with open(temp_filename, "rb") as af2:
            params2 = dict(trans_params_base, file=af2)
            eng_response = client.audio.translations.create(**params2)
            english_text = eng_response.text

    transcribed_text = english_text if english_text else native_text
    
    # Audit logic
    system_prompt = """
    You are an Expert Quality & Compliance Auditor for GoBumpr Customer Care (formerly MyTVS).
    Your goal is to evaluate the performance of the Customer Care Service Agent ONLY based on their spoken side of the call.
    STRICT RULES:
    1. YOU ARE REVIEWING THE AGENT'S AUDIO ONLY. 
    2. DO NOT comment on the customer.
    CRITICAL SECURITY STEP: Redact all PII. Replace with [REDACTED].
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
        "redacted_transcript": "AGENT TRANSCRIPT WITH [REDACTED] TAGS",
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
            {"role": "user", "content": f"Transcript:\\n\\n{transcribed_text}"}
        ],
        response_format={ "type": "json_object" }
    )
    
    audit_data = json.loads(audit_response.choices[0].message.content)

    # Diarization
    diarized_segments = []
    raw_segments = getattr(transcript_response, "segments", [])
    if raw_segments:
        segments_text = "\\n".join(
            f"[{i}] ({round(s.get('start', 0), 1)}s) {s.get('text', '').strip()}"
            for i, s in enumerate(raw_segments)
        )
        diarize_prompt = f"""You are analyzing a customer care phone call transcript. There are exactly 2 speakers: AGENT (customer care representative) and CUSTOMER (caller).
Return ONLY valid JSON: {{"segments": [{{"id": 0, "speaker": "AGENT", "text": "..."}}]}}
Segments:
{segments_text}"""
        try:
            diarize_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": diarize_prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            diarized_segments = json.loads(diarize_response.choices[0].message.content).get("segments", [])
        except:
            pass

    return {
        "text": audit_data.get("redacted_transcript", transcribed_text),
        "native_text": native_text,
        "diarized_segments": diarized_segments,
        "audit": audit_data,
        "detected_language_code": detected_lang_code,
    }

@app.callback(
    Output('file-status', 'children'),
    Input('upload-audio', 'filename')
)
def update_file_status(filename):
    if filename:
        return f"📁 {filename}"
    return dash.no_update

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
        Output('audio-player', 'src')
    ],
    Input('start-btn', 'n_clicks'),
    State('upload-audio', 'contents'),
    State('upload-audio', 'filename'),
    State('lang-select', 'value'),
    prevent_initial_call=True
)
def run_analysis(n_clicks, contents, filename, lang):
    if not contents:
        return ["--", html.Div("Please upload audio first.", className="italic"), "--", html.Div("No risks detected...", className="italic text-gray-400")] + [dash.no_update]*6

    # Decode and save audio safely
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    temp_filename = f"dash_temp_{int(time.time())}.mp3"
    
    try:
        with open(temp_filename, "wb") as f:
            f.write(decoded)
            
        data = process_audio_file(temp_filename, lang)
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
                    
        # Tone Chart
        emotions = audit.get('emotion_scores', {})
        labels = list(emotions.keys())
        values = list(emotions.values())
        colors = ['#ef4444', '#6366f1', '#a855f7', '#fb923c', '#22c55e', '#64748b', '#3b82f6', '#fbbf24']
        
        fig_tone = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
        fig_tone.update_layout(margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig_tone.update_xaxes(showgrid=False)
        fig_tone.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
        
        dom_emotion = max(emotions, key=emotions.get).upper() if emotions else "UNKNOWN"
        
        # Journey Chart
        journey = audit.get('sentiment_journey', [])
        fig_journey = go.Figure(data=[go.Scatter(y=journey, mode='lines+markers', line=dict(color='#22c55e', width=3), fill='tozeroy', fillcolor='rgba(34,197,94,0.1)')])
        fig_journey.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig_journey.update_xaxes(showgrid=False, showticklabels=False)
        fig_journey.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.05)', range=[-1.1, 1.1])
        
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
        
        return quality_score, criteria_list, acc, risk_ui, fig_tone, dom_emotion, fig_journey, transcript_ui, "mb-4 block", contents
        
    except Exception as e:
        print(e)
        return ["Error", html.Div(str(e)), "--", html.Div("Error evaluating", className="italic text-red-500")] + [dash.no_update]*6
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == '__main__':
    app.run(debug=True, port=8050)
