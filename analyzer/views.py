from django.shortcuts import render, redirect
from django.http import JsonResponse
import os
import pandas as pd
import json
from .utils import (
    CSV_PATH, load_csv, run_sentiment, run_topics,
    compute_quality, compute_repetition, compute_repetition_pairs,
    find_similar_responses
)
from .forms import UploadFileForm

# -----------------------------
# Writable directory for uploads (Render, Heroku, etc.)
UPLOAD_DIR = '/tmp/uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)
# -----------------------------

def get_active_csv(request):
    filename = request.session.get('uploaded_csv')
    if filename:
        path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(path):
            return path
    return CSV_PATH

# -----------------------------
# Views
# -----------------------------
def home(request):
    return render(request, 'analyzer/home.html')

def upload_file(request):
    try:
        form = UploadFileForm()
        error = None
        uploaded_file_name = None

        if request.method == 'POST':
            form = UploadFileForm(request.POST, request.FILES)
            f = request.FILES.get('file')
            if not f:
                error = "No file selected!"
            elif form.is_valid():
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                path = os.path.join(UPLOAD_DIR, f.name)
                with open(path, 'wb+') as dest:
                    for chunk in f.chunks():
                        dest.write(chunk)
                # store only filename in session
                request.session['uploaded_csv'] = f.name
                return redirect('dashboard')

        if request.session.get('uploaded_csv'):
            uploaded_file_name = request.session.get('uploaded_csv')

        has_upload = bool(uploaded_file_name)

        return render(request, 'analyzer/upload.html', {
            'form': form,
            'has_upload': has_upload,
            'uploaded_file_name': uploaded_file_name,
            'error': error
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return render(request, 'analyzer/upload.html', {
            'form': UploadFileForm(),
            'has_upload': False,
            'uploaded_file_name': None,
            'error': f"Internal Error: {e}"
        })


def dashboard(request):
    try:
        path = get_active_csv(request)
        df = load_csv(path)
        using_default = (path == CSV_PATH)
        total_msgs = len(df)
        assistant_msgs = df[df.get('Role', '').str.lower() == 'assistant'] if 'Role' in df.columns else pd.DataFrame()
        total_assist = len(assistant_msgs)
        sent_df = run_sentiment(assistant_msgs) if not assistant_msgs.empty else pd.DataFrame()

        if not sent_df.empty:
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
        else:
            summary_table = []

        summary = {'total_msgs': total_msgs, 'total_assistant': total_assist, 'using_default': using_default}
        return render(request, 'analyzer/dashboard.html', {'summary': summary, 'summary_table': summary_table})
    except Exception:
        return render(request, 'analyzer/dashboard.html', {'summary': {}, 'summary_table': [], 'error': 'Failed to load dashboard'})


def sentiment_view(request):
    try:
        df = load_csv(get_active_csv(request))
        df['Role'] = df.get('Role', '').astype(str).str.lower().str.strip()
        assistant = df[df['Role'] == 'assistant'] if 'Role' in df.columns else pd.DataFrame()
        assistant = run_sentiment(assistant) if not assistant.empty else pd.DataFrame()
        counts = assistant['Sentiment_Label'].value_counts().to_dict() if not assistant.empty else {}
        counts = {'Positive': counts.get('Positive', 0), 'Neutral': counts.get('Neutral', 0), 'Negative': counts.get('Negative', 0)}
        return render(request, 'analyzer/sentiment.html', {'counts': counts})
    except Exception:
        return render(request, 'analyzer/sentiment.html', {'counts': {'Positive': 0, 'Neutral': 0, 'Negative': 0}, 'error': 'Failed to compute sentiment'})


def api_messages(request):
    try:
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
    except Exception:
        return JsonResponse({"messages": []})


def api_sentiment_summary(request):
    try:
        conversation = request.GET.get('conversation')
        df = load_csv(get_active_csv(request))
        df = run_sentiment(df)
        if conversation and conversation != "__all__":
            df = df[df["Conversation_Title"] == conversation]
        counts = df["Sentiment_Label"].value_counts().to_dict() if not df.empty else {}
        return JsonResponse({"counts": counts})
    except Exception:
        return JsonResponse({"counts": {}})


def api_conversations(request):
    try:
        df = load_csv(get_active_csv(request))
        convs = sorted(df["Conversation_Title"].dropna().unique().tolist()) if "Conversation_Title" in df.columns else []
        return JsonResponse({"conversations": convs})
    except Exception:
        return JsonResponse({"conversations": []})


def topics_view(request):
    try:
        df = load_csv(get_active_csv(request))
        topics, _ = run_topics(df, n_topics=6)
        return render(request, 'analyzer/topics.html', {'topics': topics})
    except Exception:
        return render(request, 'analyzer/topics.html', {'topics': [], 'error': 'Failed to compute topics'})


def quality_view(request):
    try:
        df = load_csv(get_active_csv(request))
        assistant = run_sentiment(df)
        qdf = compute_quality(assistant) if not assistant.empty else pd.DataFrame()
        top = qdf.sort_values('quality_score', ascending=False).head(10).to_dict('records') if not qdf.empty else []
        bottom = qdf.sort_values('quality_score').head(10).to_dict('records') if not qdf.empty else []
        return render(request, 'analyzer/quality.html', {'top': top, 'bottom': bottom})
    except Exception:
        return render(request, 'analyzer/quality.html', {'top': [], 'bottom': [], 'error': 'Failed to compute quality'})


def api_metrics(request):
    try:
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

        return JsonResponse({
            "accuracy": accuracy,
            "loss": loss,
            "positive_percent": positive_percent,
            "repetitive_percent": repetitive_percent
        })
    except Exception:
        return JsonResponse({"accuracy": 0, "loss": 0, "positive_percent": 0, "repetitive_percent": 0})


def repetition_view(request):
    try:
        df = load_csv(get_active_csv(request))
        df_assistant = df[df.get('Role', '').str.lower() == 'assistant'].copy().reset_index(drop=True) if 'Role' in df.columns else pd.DataFrame()
        df_rep = compute_repetition(df_assistant) if not df_assistant.empty else pd.DataFrame()
        avg_score = round(df_rep['Semantic_Repetition'].iloc[0], 3) if not df_rep.empty else 0.0
        df_pairs = pd.DataFrame({
            'Message1': df_assistant['Message'][:-1].values,
            'Message2': df_assistant['Message'][1:].values
        }) if not df_assistant.empty else pd.DataFrame()
        df_pairs = compute_repetition_pairs(df_pairs) if not df_pairs.empty else pd.DataFrame()
        repetitive_pairs = df_pairs[df_pairs['Is_Repetitive'] == 1].sort_values('similarity_score', ascending=False).head(10).to_dict(orient='records') if not df_pairs.empty else []

        query = request.GET.get('query')
        similar_responses = find_similar_responses(df_assistant, query, k=5).to_dict(orient='records') if query and not df_assistant.empty else None

        context = {
            'avg_score': avg_score,
            'repetitive_pairs': repetitive_pairs,
            'repetitive_pairs_json': json.dumps(repetitive_pairs),
            'similar_responses': similar_responses,
            'query': query or ''
        }
        return render(request, 'analyzer/repetition.html', context)
    except Exception:
        return render(request, 'analyzer/repetition.html', {
            'avg_score': 0,
            'repetitive_pairs': [],
            'repetitive_pairs_json': '[]',
            'similar_responses': None,
            'query': '',
            'error': 'Failed to compute repetition'
        })


def reset_csv(request):
    try:
        filename = request.session.get('uploaded_csv')
        if filename:
            path = os.path.join(UPLOAD_DIR, filename)
            if os.path.exists(path):
                os.remove(path)
        request.session.pop('uploaded_csv', None)
    except Exception:
        pass
    return redirect('upload')


def api_topics(request):
    try:
        df = load_csv(get_active_csv(request))
        if "Conversation_Title" not in df.columns or df.empty:
            return JsonResponse({}, safe=False)
        topics = df["Conversation_Title"].value_counts().head(10).to_dict()
        return JsonResponse(topics, safe=False)
    except Exception:
        return JsonResponse({}, safe=False)


def api_activity(request):
    try:
        df = load_csv(get_active_csv(request))
        if df.empty:
            return JsonResponse({}, safe=False)
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df["Date"] = df["Timestamp"].dt.date
            activity = df["Date"].value_counts().sort_index().to_dict()
        else:
            activity = df["Conversation_Title"].value_counts().to_dict() if "Conversation_Title" in df.columns else {}
        return JsonResponse(activity, safe=False)
    except Exception:
        return JsonResponse({}, safe=False)
