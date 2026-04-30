from __future__ import annotations

import argparse
import html
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from data_utils import prepare_inference_dataframe
from model import TwoStageABSAModel


DEFAULT_MODEL_PATH = Path("artifacts/absa_model.joblib")


def clean_optional(value: str) -> str | None:
    text = value.strip()
    return text or None


def predict_single_review(
    model: TwoStageABSAModel,
    review_text: str,
    star_rating: str | None = None,
    business_category: str | None = None,
    platform: str | None = None,
    business_name: str | None = None,
) -> dict[str, Any]:
    df = prepare_inference_dataframe(
        [
            {
                "review_id": 1,
                "review_text": review_text,
                "star_rating": clean_optional(star_rating or "") or None,
                "business_category": clean_optional(business_category or "") or None,
                "platform": clean_optional(platform or "") or None,
                "business_name": clean_optional(business_name or "") or None,
            }
        ]
    )

    record = model.predict_records(df)[0]
    probability_row = model.predict_aspect_probabilities(df).iloc[0]

    aspect_rows: list[dict[str, Any]] = []
    for aspect in model.aspects:
        probability = float(probability_row[aspect])
        threshold = float(model.aspect_thresholds.get(aspect, 0.5))
        aspect_rows.append(
            {
                "aspect": aspect,
                "probability": probability,
                "threshold": threshold,
                "selected": aspect in record["aspects"],
                "sentiment": record["aspect_sentiments"].get(aspect, ""),
            }
        )
    aspect_rows.sort(key=lambda item: item["probability"], reverse=True)

    return {
        "normalized_review_text": df.iloc[0]["normalized_review_text"],
        "prediction": record,
        "aspect_rows": aspect_rows,
    }


def render_page(
    form_values: dict[str, str],
    result: dict[str, Any] | None = None,
    error_message: str | None = None,
    model_path: Path | None = None,
) -> str:
    review_text = html.escape(form_values.get("review_text", ""))
    business_name = html.escape(form_values.get("business_name", ""))
    business_category = html.escape(form_values.get("business_category", ""))
    platform = html.escape(form_values.get("platform", ""))
    star_rating = html.escape(form_values.get("star_rating", ""))

    result_html = ""
    if error_message:
        result_html = f"""
        <section class="card error">
          <h2>Error</h2>
          <p>{html.escape(error_message)}</p>
        </section>
        """
    elif result is not None:
        prediction = result["prediction"]
        prediction_json = html.escape(json.dumps(prediction, ensure_ascii=False, indent=2))
        normalized_text = html.escape(result["normalized_review_text"])
        aspect_items = "".join(
            f"<span class='chip'>{html.escape(aspect)}: {html.escape(sentiment)}</span>"
            for aspect, sentiment in prediction["aspect_sentiments"].items()
        )
        aspect_table = "".join(
            (
                "<tr class='{row_class}'>"
                "<td>{aspect}</td>"
                "<td>{probability:.4f}</td>"
                "<td>{threshold:.2f}</td>"
                "<td>{selected}</td>"
                "<td>{sentiment}</td>"
                "</tr>"
            ).format(
                row_class="selected" if row["selected"] else "",
                aspect=html.escape(row["aspect"]),
                probability=row["probability"],
                threshold=row["threshold"],
                selected="yes" if row["selected"] else "no",
                sentiment=html.escape(row["sentiment"] or "-"),
            )
            for row in result["aspect_rows"]
        )

        result_html = f"""
        <section class="card">
          <h2>Prediction</h2>
          <div class="chips">{aspect_items or "<span class='chip'>no prediction</span>"}</div>
          <div class="result-grid">
            <div>
              <h3>Normalized Review</h3>
              <p class="review-box" dir="rtl">{normalized_text}</p>
            </div>
            <div>
              <h3>Submission JSON</h3>
              <pre>{prediction_json}</pre>
            </div>
          </div>
          <h3>Aspect Scores</h3>
          <table>
            <thead>
              <tr>
                <th>Aspect</th>
                <th>Probability</th>
                <th>Threshold</th>
                <th>Selected</th>
                <th>Sentiment</th>
              </tr>
            </thead>
            <tbody>
              {aspect_table}
            </tbody>
          </table>
        </section>
        """

    model_path_text = html.escape(str(model_path or DEFAULT_MODEL_PATH))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Arabic ABSA Demo</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f1ea;
      --paper: #fffdf9;
      --ink: #202124;
      --muted: #5f6368;
      --line: #d6cfc3;
      --accent: #0b6e4f;
      --accent-soft: #e5f3ee;
      --error: #8a1c1c;
      --error-soft: #fdeaea;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, #efe1cd 0, transparent 30%),
        radial-gradient(circle at top right, #dae9e2 0, transparent 35%),
        var(--bg);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      margin-bottom: 24px;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px;
      line-height: 1.15;
    }}
    h1 {{
      font-size: 2.2rem;
    }}
    p {{
      margin: 0 0 12px;
      color: var(--muted);
    }}
    .card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
      margin-bottom: 18px;
    }}
    .error {{
      background: var(--error-soft);
      border-color: #efc7c7;
      color: var(--error);
    }}
    .grid {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    label {{
      display: block;
      margin-bottom: 6px;
      font-size: 0.95rem;
      color: var(--ink);
    }}
    input, textarea, select, button {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      padding: 12px 14px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }}
    textarea {{
      min-height: 180px;
      resize: vertical;
      direction: rtl;
    }}
    button {{
      cursor: pointer;
      background: var(--accent);
      color: #fff;
      border: none;
      font-weight: 600;
      margin-top: 4px;
    }}
    button:hover {{
      filter: brightness(1.04);
    }}
    .meta {{
      display: inline-block;
      background: var(--accent-soft);
      color: var(--accent);
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 0.9rem;
      margin-bottom: 12px;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 16px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 600;
    }}
    .result-grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      margin-bottom: 16px;
    }}
    .review-box, pre {{
      margin: 0;
      padding: 14px;
      border-radius: 12px;
      background: #fcfaf6;
      border: 1px solid var(--line);
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 14px;
      border: 1px solid var(--line);
    }}
    th, td {{
      padding: 12px 10px;
      text-align: left;
      border-bottom: 1px solid var(--line);
    }}
    th {{
      background: #f2ede4;
    }}
    tr.selected {{
      background: #edf7f2;
    }}
    .footer {{
      font-size: 0.92rem;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <span class="meta">Local ABSA Demo</span>
      <h1>Arabic Aspect-Based Sentiment Analysis</h1>
      <p>Paste one review, click predict, and the page will show the detected aspects, sentiment for each aspect, and the exact JSON result format.</p>
      <p class="footer">Loaded model: <code>{model_path_text}</code></p>
    </section>

    <section class="card">
      <h2>Try a Review</h2>
      <form method="post" action="/predict">
        <label for="review_text">Review text</label>
        <textarea id="review_text" name="review_text" placeholder="اكتب المراجعة هنا" required>{review_text}</textarea>

        <div class="grid">
          <div>
            <label for="star_rating">Star rating (optional)</label>
            <select id="star_rating" name="star_rating">
              <option value="" {"selected" if not star_rating else ""}>Unknown</option>
              <option value="1" {"selected" if star_rating == "1" else ""}>1</option>
              <option value="2" {"selected" if star_rating == "2" else ""}>2</option>
              <option value="3" {"selected" if star_rating == "3" else ""}>3</option>
              <option value="4" {"selected" if star_rating == "4" else ""}>4</option>
              <option value="5" {"selected" if star_rating == "5" else ""}>5</option>
            </select>
          </div>
          <div>
            <label for="business_category">Business category (optional)</label>
            <input id="business_category" name="business_category" value="{business_category}" placeholder="restaurant / ecommerce / healthcare">
          </div>
          <div>
            <label for="platform">Platform (optional)</label>
            <input id="platform" name="platform" value="{platform}" placeholder="google_maps / play_store">
          </div>
          <div>
            <label for="business_name">Business name (optional)</label>
            <input id="business_name" name="business_name" value="{business_name}" placeholder="Store or app name">
          </div>
        </div>

        <button type="submit">Predict</button>
      </form>
    </section>

    {result_html}
  </main>
</body>
</html>
"""


class ABSARequestHandler(BaseHTTPRequestHandler):
    model: TwoStageABSAModel | None = None
    model_path: Path = DEFAULT_MODEL_PATH

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            payload = json.dumps({"status": "ok"}, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if parsed.path != "/":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        page = render_page({}, model_path=self.model_path)
        self._send_html(page)

    def do_POST(self) -> None:
        if self.path != "/predict":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        form_values = self._read_form_values()
        review_text = form_values.get("review_text", "").strip()
        if not review_text:
            page = render_page(form_values, error_message="Review text is required.", model_path=self.model_path)
            self._send_html(page, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            result = predict_single_review(
                model=self._require_model(),
                review_text=review_text,
                star_rating=form_values.get("star_rating"),
                business_category=form_values.get("business_category"),
                platform=form_values.get("platform"),
                business_name=form_values.get("business_name"),
            )
            page = render_page(form_values, result=result, model_path=self.model_path)
            self._send_html(page)
        except Exception as exc:
            page = render_page(form_values, error_message=str(exc), model_path=self.model_path)
            self._send_html(page, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_form_values(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        parsed = parse_qs(body, keep_blank_values=True)
        return {key: values[0] if values else "" for key, values in parsed.items()}

    def _send_html(self, page: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = page.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    @classmethod
    def _require_model(cls) -> TwoStageABSAModel:
        if cls.model is None:
            raise ValueError(f"Model not loaded. Expected artifact at: {cls.model_path}")
        return cls.model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a lightweight local app for Arabic ABSA predictions.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the trained model artifact.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the local server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the local server.")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ABSARequestHandler.model_path = args.model_path
    ABSARequestHandler.model = TwoStageABSAModel.load(args.model_path)

    server = ThreadingHTTPServer((args.host, args.port), ABSARequestHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"ABSA app running at {url}")

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
