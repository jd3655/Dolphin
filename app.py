import json
import logging
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Optional

import gradio as gr

from dolphin_app.parser import parse_document


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dolphin_gui")


SUPPORTED_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".pdf"]


def _copy_uploads(files: List[gr.File]) -> List[Path]:
    """Copy uploaded files into a temp directory with stable names."""
    run_dir = Path(tempfile.mkdtemp(prefix="dolphin_gui_uploads_"))
    copied_paths: List[Path] = []
    for f in files:
        src = Path(f.name)
        dest = run_dir / src.name
        shutil.copy(src, dest)
        copied_paths.append(dest)
    return copied_paths


def _build_markdown(results: List[dict]) -> str:
    sections = []
    for res in results:
        heading = f"## {res['input_name']}"
        body = res.get("markdown") or "_Markdown output not available._"
        sections.append(f"{heading}\n\n{body}")
    return "\n\n---\n\n".join(sections)


def _build_json_preview(results: List[dict]) -> str:
    payload = [
        {"input_name": res["input_name"], "json": res.get("json")}
        for res in results
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _zip_outputs(output_paths: List[str], save_dir: Path) -> Optional[str]:
    if not output_paths:
        return None
    zip_dir = save_dir / "gui_exports"
    zip_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    zip_path = zip_dir / f"dolphin_outputs_{timestamp}.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for path_str in output_paths:
            path = Path(path_str)
            if path.exists():
                zipf.write(path, arcname=path.name)
    return str(zip_path)


def run_parsing(
    files: List[gr.File],
    model_path: str,
    output_dir: str,
    max_batch_size: int,
    combine_output: bool,
    progress=gr.Progress(track_tqdm=True),
):
    if not files:
        raise gr.Error("Please upload at least one image or PDF.")

    logs = []

    def log(msg: str):
        logs.append(msg)
        logger.info(msg)

    log(f"Received {len(files)} file(s). Copying to a temp directory...")
    copied_files = _copy_uploads(files)
    results = []
    all_outputs: List[str] = []

    try:
        total = len(copied_files)
        for idx, path in enumerate(copied_files, start=1):
            if path.suffix.lower() not in SUPPORTED_EXTS:
                raise gr.Error(f"Unsupported file type for {path.name}. Supported: {', '.join(SUPPORTED_EXTS)}")
            log(f"[{idx}/{total}] Processing {path.name}")
            progress((idx - 1) / total, f"Processing {path.name}")
            res = parse_document(
                input_path=path,
                model_path=model_path,
                save_dir=output_dir,
                max_batch_size=max_batch_size,
                device=None,
            )
            results.append(res)
            all_outputs.extend(res.get("output_files", []))
        progress(1.0, "Completed")
    finally:
        shutil.rmtree(copied_files[0].parent, ignore_errors=True)

    if not results:
        raise gr.Error("No files were processed.")

    markdown_preview = _build_markdown(results) if combine_output else results[-1]["markdown"]
    json_preview = _build_json_preview(results) if combine_output else results[-1]["json_text"]
    zip_path = _zip_outputs(all_outputs, Path(output_dir))

    return (
        markdown_preview,
        json_preview,
        "\n".join(logs),
        all_outputs,
        zip_path,
    )


with gr.Blocks(title="Dolphin Document Parser") as demo:
    gr.Markdown(
        "# Dolphin Document Parser\n"
        "Drag and drop images or PDFs to parse them with the Dolphin model. "
        "Results are saved to the output directory and previewed below."
    )
    with gr.Row():
        with gr.Column(scale=1):
            files_input = gr.File(
                label="Upload images or PDFs",
                file_types=SUPPORTED_EXTS,
                file_count="multiple",
                type="file",
            )
            model_path_input = gr.Textbox(
                label="Model path",
                value="./hf_model",
                placeholder="Path to model (local directory)",
            )
            output_dir_input = gr.Textbox(
                label="Output directory",
                value="./results_gui",
            )
            max_batch_slider = gr.Slider(
                label="Max batch size",
                minimum=1,
                maximum=16,
                step=1,
                value=8,
            )
            combine_checkbox = gr.Checkbox(
                label="Return combined output for multi-file runs",
                value=True,
            )
            run_button = gr.Button("Parse", variant="primary")

        with gr.Column(scale=1):
            with gr.Tab("Markdown"):
                markdown_output = gr.Markdown(label="Markdown preview")
            with gr.Tab("JSON"):
                json_output = gr.Code(label="JSON preview", language="json", interactive=False)
            with gr.Tab("Logs"):
                logs_output = gr.Textbox(label="Logs", lines=12)

            downloads = gr.Files(label="Generated files")
            zip_download = gr.File(label="Download all as zip")

    run_button.click(
        fn=run_parsing,
        inputs=[files_input, model_path_input, output_dir_input, max_batch_slider, combine_checkbox],
        outputs=[markdown_output, json_output, logs_output, downloads, zip_download],
        queue=True,
    )

demo.queue(concurrency_count=1)


if __name__ == "__main__":
    demo.launch()
