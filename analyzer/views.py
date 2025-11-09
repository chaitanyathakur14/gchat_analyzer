from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
import os
import pandas as pd
import json
from .utils import (
    CSV_PATH, load_csv, run_sentiment, run_topics,
    compute_quality, compute_repetition, compute_repetition_pairs,
    find_similar_responses
)
from .forms import UploadFileForm

UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_active_csv(request):
    path = request.session.get('uploaded_csv')
    if path and os.path.exists(path):
        return path
    return CSV_PATH

def home(request):
    return render(request, 'analyzer/home.html')

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            f = request.FILES['file']
            path = os.path.join(UPLOAD_DIR, f.name)
            with open(path, 'wb+') as dest:
                for chunk in f.chunks():
                    dest.write(chunk)
            request.session['uploaded_csv'] = path
            return redirect('dashboard')
    else:
        form = UploadFileForm()
    has_upload = bool(request.session.get('uploaded_csv'))
    return render(request, 'analyzer/upload.html', {'form': form, 'has_upload': has_upload})

def dashboard(request):
    path = get_active_csv(request)
    df = load_csv(path)
    using_default = (path == CSV_PATH)
    total_msgs = len(df)
    assistant_msgs = df[df['Role'].str.lower() == 'assistant']
    total_assist = len(assistant_msgs)
    sent_df = run_sentiment(assistant_msgs)
    conv_summary = (
        sent_df.groupby("Conversation_Title")["Sentiment_Label"]
        .value_counts(normalize=True)
        .mul(100)
        .unstack(fill_value=0)
        .reset_index()
    )
    conv_summary["Total_Messages"] = sent_df.groupby("Conversation_Title").size().values
    conv_summary = conv_summary.rename(columns={
        "Conversation_Title": "Conversation",
        "Positive": "Positive %",
        "Negative": "Negative %",
        "Neutral": "Neutral %"
    })
    summary_table = conv_summary.to_dict(orient="records")
    summary = {'total_msgs': total_msgs, 'total_assistant': total_assist, 'using_default': using_default}
    return render(request, 'analyzer/dashboard.html', {'summary': summary, 'summary_table': summary_table})

def sentiment_view(request):
    df = load_csv(get_active_csv(request))
    df['Role'] = df['Role'].astype(str).str.lower().str.strip()
    assistant = df[df['Role'] == 'assistant']
    assistant = run_sentiment(assistant)
    counts = assistant['Sentiment_Label'].value_counts().to_dict()
    counts = {'Positive': counts.get('Positive', 0), 'Neutral': counts.get('Neutral', 0), 'Negative': counts.get('Negative', 0)}
    return render(request, 'analyzer/sentiment.html', {'counts': counts})

def api_messages(request):
    q = request.GET.get('q', '').lower()
    sentiment = request.GET.get('sentiment')
    conversation = request.GET.get('conversation')
    limit = int(request.GET.get('limit', 100))
    df = load_csv(get_active_csv(request))
    df = run_sentiment(df)
    required_cols = ["Role", "Conversation_Title", "Message", "Sentiment_Label"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""
    df = df.fillna("")
    if conversation and conversation != "__all__":
        df = df[df["Conversation_Title"] == conversation]
    if sentiment:
        df = df[df["Sentiment_Label"].str.lower() == sentiment.lower()]
    if q:
        df = df[df["Message"].str.lower().str.contains(q)]
    records = df.head(limit)[required_cols].replace({pd.NA: "", None: ""}).to_dict("records")
    return JsonResponse({"messages": records})

def api_sentiment_summary(request):
    conversation = request.GET.get('conversation')
    df = load_csv(get_active_csv(request))
    df = run_sentiment(df)
    if conversation and conversation != "__all__":
        df = df[df["Conversation_Title"] == conversation]
    counts = df["Sentiment_Label"].value_counts().to_dict()
    return JsonResponse({"counts": counts})

def api_conversations(request):
    df = load_csv(get_active_csv(request))
    convs = sorted(df["Conversation_Title"].dropna().unique().tolist())
    return JsonResponse({"conversations": convs})

def topics_view(request):
    df = load_csv(get_active_csv(request))
    topics, _ = run_topics(df, n_topics=6)
    return render(request, 'analyzer/topics.html', {'topics': topics})

def quality_view(request):
    df = load_csv(get_active_csv(request))
    assistant = run_sentiment(df)
    qdf = compute_quality(assistant)
    top = qdf.sort_values('quality_score', ascending=False).head(10).to_dict('records')
    bottom = qdf.sort_values('quality_score').head(10).to_dict('records')
    return render(request, 'analyzer/quality.html', {'top': top, 'bottom': bottom})

def api_metrics(request):
    conv = request.GET.get("conversation")
    df = load_csv(get_active_csv(request))
    df = run_sentiment(df)
    if conv and conv != "__all__":
        df = df[df["Conversation_Title"] == conv]
    if df.empty:
        return JsonResponse({"accuracy": 0, "loss": 0, "positive_percent": 0, "repetitive_percent": 0})
    total = len(df)
    pos = (df["Sentiment_Label"] == "Positive").sum()
    positive_percent = round((pos / total) * 100, 2) if total > 0 else 0
    rep_df = compute_repetition(df)
    repetitive_percent = float(rep_df["Semantic_Repetition"].mean()) if "Semantic_Repetition" in rep_df.columns else 0.0
    accuracy = round(80 + (positive_percent / 10), 2)
    loss = round(1 - accuracy / 100, 3)
    return JsonResponse({"accuracy": accuracy, "loss": loss, "positive_percent": positive_percent, "repetitive_percent": repetitive_percent})

def repetition_view(request):
    df = load_csv(get_active_csv(request))
    df_assistant = df[df['Role'].str.lower() == 'assistant'].copy().reset_index(drop=True)
    df_rep = compute_repetition(df_assistant)
    avg_score = round(df_rep['Semantic_Repetition'].iloc[0], 3) if not df_rep.empty else 0.0
    df_pairs = pd.DataFrame({'Message1': df_assistant['Message'][:-1].values, 'Message2': df_assistant['Message'][1:].values})
    df_pairs = compute_repetition_pairs(df_pairs)
    repetitive_pairs = df_pairs[df_pairs['Is_Repetitive'] == 1].sort_values('similarity_score', ascending=False).head(10).to_dict(orient='records')
    query = request.GET.get('query')
    similar_responses = None
    if query:
        similar_responses = find_similar_responses(df_assistant, query, k=5).to_dict(orient='records')
    repetitive_pairs_json = json.dumps(repetitive_pairs)
    context = {'avg_score': avg_score, 'repetitive_pairs': repetitive_pairs, 'repetitive_pairs_json': repetitive_pairs_json, 'similar_responses': similar_responses, 'query': query or ''}
    return render(request, 'analyzer/repetition.html', context)

def reset_csv(request):
    path = request.session.get('uploaded_csv')
    if path and os.path.exists(path):
        os.remove(path)
    request.session.pop('uploaded_csv', None)
    return redirect('upload')

def api_topics(request):
    df = load_csv(get_active_csv(request))
    try:
        if "Conversation_Title" not in df.columns or df.empty:
            return JsonResponse({}, safe=False)
        topics = df["Conversation_Title"].value_counts().head(10).to_dict()
        return JsonResponse(topics, safe=False)
    except:
        return JsonResponse({}, status=500)

def api_activity(request):
    df = load_csv(get_active_csv(request))
    try:
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df["Date"] = df["Timestamp"].dt.date
            activity = df["Date"].value_counts().sort_index().to_dict()
        else:
            activity = df["Conversation_Title"].value_counts().to_dict()
        return JsonResponse(activity, safe=False)
    except:
        return JsonResponse({}, status=500)
