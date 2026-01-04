import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from demo_page import DOLPHIN, process_document
from utils.utils import setup_output_dirs

logger = logging.getLogger(__name__)


def _device_key(device: Optional[str]) -> str:
    return device or "auto"


@lru_cache(maxsize=2)
def _load_model_cached(model_path: str, device: str) -> DOLPHIN:
    """Load and cache the model to avoid repeated initialization."""
    logger.info("Loading model from %s (device=%s)", model_path, device)
    chosen_device: Optional[str] = None if device == "auto" else device
    return DOLPHIN(model_path, device=chosen_device)


def get_model(model_path: Path, device: Optional[str] = None) -> DOLPHIN:
    """Return a cached model instance."""
    resolved_path = str(model_path.expanduser().resolve())
    return _load_model_cached(resolved_path, _device_key(device))


def _read_markdown(save_dir: Path, base_name: str) -> Optional[str]:
    md_path = save_dir / "markdown" / f"{base_name}.md"
    if md_path.exists():
        return md_path.read_text(encoding="utf-8")
    return None


def _collect_outputs(save_dir: Path, base_name: str) -> List[Path]:
    outputs: List[Path] = []
    for candidate in [
        save_dir / "markdown" / f"{base_name}.md",
        save_dir / "output_json" / f"{base_name}.json",
        save_dir / "recognition_json" / f"{base_name}.json",
        save_dir / "layout_visualization" / f"{base_name}_layout.png",
    ]:
        if candidate.exists():
            outputs.append(candidate)

    figures_dir = save_dir / "markdown" / "figures"
    if figures_dir.exists():
        outputs.extend(sorted(figures_dir.glob(f"{base_name}_figure_*.png")))

    return outputs


def parse_document(
    input_path: str | Path,
    model_path: str | Path,
    save_dir: str | Path,
    max_batch_size: int = 8,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse a document (image or PDF) and return structured results.

    Returns:
        dict with keys:
            - input_name: original filename
            - markdown: rendered markdown text (if generated)
            - json: structured results (list/dict)
            - json_text: pretty-printed JSON string
            - output_files: list of output file paths written to disk
            - json_path: path to the saved JSON file (if any)
            - markdown_path: path to the saved markdown file (if any)
    """

    path = Path(input_path)
    save_dir_path = Path(save_dir)
    model_path_obj = Path(model_path)

    setup_output_dirs(str(save_dir_path))

    model = get_model(model_path_obj, device=device)
    json_path, recognition_results = process_document(
        document_path=str(path),
        model=model,
        save_dir=str(save_dir_path),
        max_batch_size=max_batch_size,
    )

    base_name = path.stem

    markdown_content = _read_markdown(save_dir_path, base_name)
    json_text: Optional[str] = None
    if json_path:
        try:
            json_text = Path(json_path).read_text(encoding="utf-8")
            recognition_results = json.loads(json_text)
        except Exception:
            logger.warning("Could not read JSON file at %s, using in-memory results.", json_path)

    output_files = _collect_outputs(save_dir_path, base_name)

    if markdown_content is None:
        logger.info("Markdown not found for %s. Outputs: %s", base_name, output_files)

    result = {
        "input_name": path.name,
        "markdown": markdown_content or "",
        "json": recognition_results,
        "json_text": json_text or json.dumps(recognition_results, ensure_ascii=False, indent=2),
        "output_files": [str(p) for p in output_files],
        "json_path": str(json_path) if json_path else None,
        "markdown_path": str((save_dir_path / "markdown" / f"{base_name}.md")) if markdown_content else None,
    }

    return result
