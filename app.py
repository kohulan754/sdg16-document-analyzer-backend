from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os, tempfile, traceback
from pathlib import Path

# Text extraction libs
import io
from PIL import Image
import pytesseract
import pdfplumber
import docx

# Optional OpenAI
import os
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # set this in deploy
use_openai = bool(OPENAI_API_KEY)
if use_openai:
    import openai
    openai.api_key = OPENAI_API_KEY

ALLOWED = {'.pdf','.docx','.txt','.png','.jpg','.jpeg','.tiff','.bmp'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30MB

def allowed(filename):
    return Path(filename).suffix.lower() in ALLOWED

def extract_text_from_pdf(path, ocr=False):
    text_parts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                ptext = page.extract_text()
                if ptext:
                    text_parts.append(ptext)
                elif ocr:
                    # render page as image and OCR
                    pil = page.to_image(resolution=200).original
                    text_parts.append(pytesseract.image_to_string(pil))
    except Exception:
        # fallback: try OCR on each page image
        if ocr:
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(path)
                for im in images:
                    text_parts.append(pytesseract.image_to_string(im))
            except Exception:
                pass
    return "\n\n".join([p for p in text_parts if p])

def extract_text_from_docx(path):
    doc = docx.Document(path)
    texts = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(texts)

def extract_text_from_image(path):
    img = Image.open(path)
    return pytesseract.image_to_string(img)

def simple_summary(text, max_sentences=4):
    # Very small heuristic summary fallback
    if not text:
        return "No text found."
    # split by lines and punctuation, naive approach
    import re
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    # rank by length (simple)
    sents_sorted = sorted(sents, key=lambda s: len(s), reverse=True)
    summary = " ".join(sents_sorted[:max_sentences])
    return summary[:3000]

def analyze_with_openai(text):
    # craft a direct prompt for transparency analysis
    system = "You are a civic-tech analyst. Analyze the user document for transparency, potential conflicts of interest, missing key details, signs of bias, and produce a transparency score from 0-100. Return JSON with keys: summary, transparency_score, issues (array)."
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":f"Document:\n\n{text[:30000]}"}
            ],
            max_tokens=1000,
            temperature=0.2
        )
        out = resp.choices[0].message['content']
        # The assistant should return JSON — try to parse it; if not, wrap in summary
        import json
        try:
            parsed = json.loads(out)
            return parsed
        except Exception:
            # fallback: put output into 'summary' field
            return {"summary": out, "transparency_score": None, "issues": []}
    except Exception as e:
        return {"summary": f"OpenAI call failed: {str(e)}", "transparency_score": None, "issues": []}

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({"error":"No file provided"}), 400
        f = request.files['file']
        if f.filename == '':
            return jsonify({"error":"Empty filename"}), 400
        if not allowed(f.filename):
            return jsonify({"error":"Unsupported file type"}), 400

        ocr_flag = request.form.get('ocr', '0') == '1'
        fname = secure_filename(f.filename)
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, fname)
        f.save(path)

        ext = Path(path).suffix.lower()
        extracted_text = ""

        if ext == '.pdf':
            extracted_text = extract_text_from_pdf(path, ocr=ocr_flag)
        elif ext == '.docx':
            extracted_text = extract_text_from_docx(path)
        elif ext in {'.png','.jpg','.jpeg','.tiff','.bmp'}:
            extracted_text = extract_text_from_image(path)
        elif ext == '.txt':
            with open(path, 'r', encoding='utf8', errors='ignore') as fh:
                extracted_text = fh.read()

        # Cleanup small files if needed (we keep for debug)
        metadata = {"filename": fname, "ocr_used": ocr_flag, "len_chars": len(extracted_text)}

        if not extracted_text.strip():
            # If nothing extracted and OCR disabled, try OCR as fallback
            if not ocr_flag:
                extracted_text = extract_text_from_pdf(path, ocr=True) if ext=='.pdf' else extract_text_from_image(path)
                metadata['ocr_auto'] = True

        # Run AI analysis (if enabled) otherwise fallback
        if use_openai:
            ai_result = analyze_with_openai(extracted_text)
            summary = ai_result.get('summary') if isinstance(ai_result, dict) else str(ai_result)
            score = ai_result.get('transparency_score') if isinstance(ai_result, dict) else None
            issues = ai_result.get('issues') if isinstance(ai_result, dict) else []
        else:
            summary = simple_summary(extracted_text, max_sentences=5)
            # basic heuristics for issues (very simple)
            issues = []
            low_info_phrases = ['not disclosed','no information','not provided','unknown','undisclosed']
            for p in low_info_phrases:
                if p in extracted_text.lower():
                    issues.append(f"Contains phrase indicating missing disclosure: '{p}'")
            score = max(30, min(90, 70 - (len(issues)*15)))  # heuristic

        response = {
            "summary": summary,
            "transparency_score": score,
            "issues": issues,
            "extracted_text": extracted_text,
            "metadata": metadata
        }

        return jsonify(response)

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error":"internal_server_error","detail":str(exc)}), 500

if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)), debug=True)
