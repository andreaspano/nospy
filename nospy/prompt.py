"""
Prompt builder for LLM-assisted model.json generation.

Typical usage::

    from nospy.features import FeaturesCalculator
    from nospy.prompt import generate_model_json

    calc = FeaturesCalculator(ts_df)
    new_cfg = generate_model_json(calc, model_name="nhits", h=5)
    # json/nhits.json is updated in place

Or build the prompt manually::

    from nospy.prompt import build_model_prompt

    summary = calc.summarize()
    prompt = build_model_prompt(summary, model_name="nhits", h=5)
    # pass `prompt` to any LLM
"""

import json
import os
from pathlib import Path

from nospy.config import LLMConfig

# ============================================================
# Per-model schemas
# ============================================================

_MODEL_SCHEMAS: dict[str, dict] = {
    "nhits": {
        "description": "NHITS — Neural Hierarchical Interpolation for Time Series",
        "allowed_run_params": {
            "input_size": "int — number of past observations used as input",
            "max_steps": "int — number of gradient-descent training steps",
            "learning_rate": "float — Adam learning rate",
            "batch_size": "int — number of time series sampled per training step; candidates must be ≤256",
            "windows_batch_size": "int — number of windows sampled from each series; candidates must be ≤256",
            "n_pool_kernel_size": (
                "List[int] with exactly 3 elements — "
                "max-pooling kernel size per stack (outer list is the search space, "
                "each element is a length-3 list)"
            ),
            "n_freq_downsample": (
                "List[int] with exactly 3 elements — "
                "output-frequency downsampling per stack (same shape as n_pool_kernel_size)"
            ),
            "scaler_type": 'str — one of "robust", "standard", "identity", "minmax"',
            "random_seed": "int",
            "dropout_prob_theta": "float in [0.0, 1.0]",
            "activation": 'str — one of "ReLU", "LeakyReLU", "SELU", "Tanh", "Sigmoid"',
            "mlp_units": (
                "List[List[int]] — 3 lists, one per stack, "
                "each defining the hidden-layer widths of that stack's MLP "
                "(outer list is the search space, each element is a list of 3 equal-length lists)"
            ),
            "val_check_steps": "int — validate every N training steps",
        },
        "allowed_fixed_params": {},
        "structured_examples": {
            "n_pool_kernel_size": [[1, 1, 1], [2, 2, 1], [4, 2, 1]],
            "n_freq_downsample": [[1, 1, 1], [2, 2, 1], [4, 2, 1]],
            "mlp_units": [
                [[128, 128], [128, 128], [128, 128]],
                [[256, 256], [256, 256], [256, 256]],
            ],
        },
    },
    "nbeats": {
        "description": "NBEATS — Neural Basis Expansion Analysis for Time Series",
        "allowed_run_params": {
            "input_size": "int",
            "max_steps": "int",
            "learning_rate": "float",
            "batch_size": "int — candidates must be ≤256",
            "windows_batch_size": "int — candidates must be ≤256",
            "scaler_type": 'str — one of "robust", "standard", "identity", "minmax"',
            "random_seed": "int",
        },
        "allowed_fixed_params": {
            "stack_types": (
                'List[str] — e.g. ["identity"] or ["trend", "seasonality"]. '
                "Put this in \"fixed\", NOT in \"run\"."
            ),
        },
    },
    "tft": {
        "description": "TFT — Temporal Fusion Transformer",
        "allowed_run_params": {
            "input_size": "int",
            "hidden_size": "int — model hidden dimension; must be divisible by n_head (e.g. 64, 128, 256)",
            "n_head": "int — number of attention heads; must divide hidden_size evenly (e.g. 4, 8)",
            "max_steps": "int",
            "learning_rate": "float",
            "batch_size": "int — keep ≤256; values above 256 cause crashes on typical datasets",
            "windows_batch_size": "int — keep ≤256; must not exceed total training windows (~10 × series_length)",
            "scaler_type": 'str — one of "robust", "standard", "identity", "minmax"',
            "random_seed": "int",
        },
        "allowed_fixed_params": {},
    },
}


# ============================================================
# Interpretation hints
# ============================================================


def _interpretation_hints(summary: dict) -> list[str]:
    """Derive plain-English tuning hints from a feature summary."""
    hints: list[str] = []

    ds = summary.get("dataset", {})
    level = summary.get("level", {})
    vol = summary.get("volatility", {})

    obs = ds.get("obs_count") or {}
    p50_obs = obs.get("p50", 0)
    p25_obs = obs.get("p25", 0)

    if p50_obs and p50_obs < 50:
        hints.append(
            "Series are short (median length < 50) — prefer small input_size "
            "and batch sizes to avoid overfitting."
        )
    elif p50_obs and p50_obs > 500:
        hints.append(
            "Series are long (median length > 500) — larger input_size values "
            "are feasible to capture long-range patterns. "
            "Keep batch_size ≤256 and windows_batch_size ≤ regardless of "
            "series length, to stay well within the total window count."
        )

    if p25_obs and p25_obs < 20:
        hints.append(
            f"{ds.get('pct_short_series', '?')}% of series have fewer than 20 "
            "observations — input_size should not exceed the shortest series."
        )

    trend = level.get("trend_strength") or {}
    if trend.get("p50", 0) > 0.6:
        hints.append(
            "Strong trend detected (median trend_strength > 0.6) — consider "
            "larger input_size values to give the model enough context to "
            "capture trend dynamics."
        )

    seas = level.get("seasonal_strength") or {}
    pct_seas = level.get("pct_seasonal", 0) or 0
    if seas.get("p50", 0) > 0.4 or pct_seas > 40:
        freq_dist = ds.get("frequency_distribution") or {}
        dominant_period = max(freq_dist, key=freq_dist.get) if freq_dist else "unknown"
        hints.append(
            f"Notable seasonality detected ({pct_seas}% of series, "
            f"dominant period ≈ {dominant_period}) — input_size should ideally "
            "be a multiple of the seasonal period."
        )

    entropy = level.get("entropy") or {}
    if entropy.get("p50", 0) > 0.8:
        hints.append(
            "High spectral entropy (noisy series, median > 0.8) — consider "
            "dropout regularisation and robust scaler."
        )

    arch = level.get("arch_acf") or vol.get("arch_acf") or {}
    if arch.get("count", 0) > 0 and arch.get("p50", 0) > 0.1:
        hints.append(
            "Volatility clustering detected (ARCH effects, median arch_acf > 0.1) "
            "— robust scaler is strongly recommended."
        )

    hurst = level.get("hurst") or {}
    if hurst.get("p50", 0) > 0.6:
        hints.append(
            "Long-memory / persistent series (median Hurst > 0.6) — larger "
            "input_size helps the model exploit autocorrelation structure."
        )

    zero_prop = level.get("zero_proportion") or {}
    if zero_prop.get("p50", 0) > 0.1:
        hints.append(
            "Significant zero proportion (median > 10%) — consider whether "
            "intermittent-demand handling is needed."
        )

    return hints


# ============================================================
# Public API
# ============================================================


def build_model_prompt(
    summary: dict,
    model_name: str,
    h: int,
    config=None,
    existing_json: dict | None = None,
) -> str:
    """
    Build a prompt for an LLM to generate or improve a ``model.json`` file.

    Args:
        summary:       Output of :func:`nospy.features.summarize_features`.
        model_name:    E.g. ``"nhits"``, ``"nbeats"``, ``"tft"`` (case-insensitive;
                       leading ``"auto"`` is stripped automatically).
        h:             Forecast horizon used in the experiment.
        config:        Optional :class:`~nospy.config.ExperimentConfig` or plain
                       ``dict`` with keys ``num_samples``, ``cpus``, ``gpus``.
                       Used to include compute-budget context in the prompt.
        existing_json: Current contents of the model's JSON file (if any).
                       Including it helps the LLM produce minimal, targeted diffs.

    Returns:
        A self-contained prompt string ready to be sent to an LLM.
    """
    model_key = model_name.lower().replace("auto", "")
    schema = _MODEL_SCHEMAS.get(model_key)

    lines: list[str] = []

    # ------------------------------------------------------------------ header
    lines += [
        "You are a machine learning expert specialising in neural time series "
        "forecasting with the NeuralForecast library.",
        "",
        "## Task",
        f"Generate an optimised hyperparameter search space for the **{model_name.upper()}** "
        "model as a JSON configuration file used with Ray Tune + Optuna.",
        "",
    ]

    # ----------------------------------------------------------- JSON structure
    lines += [
        "## JSON file structure",
        "```json",
        "{",
        '  "fixed": { "param": value },       // always applied, never tuned',
        '  "run":   { "param": [v1, v2, ...] }, // search space — each value must be a JSON array',
        '  "test":  { "param": value }           // single scalar values for quick smoke tests',
        "}",
        "```",
        "",
        "**Hard rules:**",
        "- `fixed` values are scalars (not arrays) — they are applied directly, never tuned",
        "- `run` values **must** be JSON arrays (they are passed to `tune.choice([...])`)",
        "- Every `run` parameter must have at least 2 candidate values",
        "- `test` values must be scalars (not arrays)",
        "- `test` section is a **smoke test only**: use the smallest valid values "
        "(e.g. `max_steps: 1`, `batch_size: 32`, `windows_batch_size: 64`, `input_size: 10`)",
        "- Use only the supported parameters listed below",
        "- Keep the correct shape for structured parameters "
        "(e.g. nested lists must preserve the documented dimensionality)",
        "- **`batch_size` candidates must all be ≤256** — larger values cause "
        "out-of-memory errors and training instability",
        "- **`windows_batch_size` candidates must all be ≤256** — the sampler "
        "crashes if this exceeds the total number of windows in the dataset "
        "(≈ n_series × (series_length − input_size − h)); 256 is safe for "
        "any dataset with ≥10 series of ≥200 observations",
        "- Return **only** valid JSON — no explanations, no markdown fences",
        "",
    ]

    # ------------------------------------------------------------- model schema
    if schema:
        lines += [f"## Model: {schema['description']}", ""]

        lines.append("### Supported `run` parameters (tuned)")
        for param, desc in schema["allowed_run_params"].items():
            lines.append(f"- `{param}`: {desc}")
        lines.append("")

        if schema.get("allowed_fixed_params"):
            lines.append(
                "### Supported `fixed` parameters (not tuned)"
            )
            for param, desc in schema["allowed_fixed_params"].items():
                lines.append(f"- `{param}`: {desc}")
            lines.append("")

        if "structured_examples" in schema:
            lines.append(
                "### Examples of valid values for structured parameters "
                "(each outer list is the search space)"
            )
            for param, examples in schema["structured_examples"].items():
                lines.append(f"- `{param}`: `{json.dumps(examples)}`")
            lines.append("")

    # --------------------------------------------------------- experiment context
    lines += ["## Experiment context", f"- Forecast horizon h = {h}"]

    if config is not None:
        if hasattr(config, "tuning"):
            tuning = config.tuning
            lines.append(f"- Tuning samples: {tuning.num_samples}")
            lines.append(f"- CPUs: {tuning.cpus}, GPUs: {tuning.gpus}")
        elif isinstance(config, dict):
            for k, v in config.items():
                lines.append(f"- {k}: {v}")

    lines.append("")

    # ---------------------------------------------------- time series summary
    lines += [
        "## Time series characteristics",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
    ]

    # ---------------------------------------------------- interpretation hints
    hints = _interpretation_hints(summary)
    if hints:
        lines.append("## Key observations for hyperparameter selection")
        for hint in hints:
            lines.append(f"- {hint}")
        lines.append("")

    # ---------------------------------------------------------- existing JSON
    if existing_json is not None:
        lines += [
            "## Current model.json (improve or replace as needed)",
            "```json",
            json.dumps(existing_json, indent=2),
            "```",
            "",
        ]

    # --------------------------------------------------------------- request
    lines += [
        "## Request",
        "Return a complete, valid model.json object with `fixed`, `run`, and `test` sections.",
        "Base your hyperparameter choices on the time series characteristics above.",
        "Return **only** the JSON object — no prose, no markdown code fences.",
    ]

    return "\n".join(lines)


# ============================================================
# GitHub Copilot API integration
# ============================================================

_COPILOT_BASE_URL = "https://api.githubcopilot.com"
_MODEL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "json"


def _get_github_token() -> str:
    """
    Resolve a GitHub token from (in order of priority):
    1. ``GITHUB_TOKEN`` environment variable
    2. ``~/.config/gh/hosts.yml`` (written by the ``gh`` CLI)

    Raises ``RuntimeError`` if no token can be found.
    """
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        return token

    hosts_file = Path.home() / ".config" / "gh" / "hosts.yml"
    if hosts_file.exists():
        import yaml  # PyYAML is already a project dependency

        hosts = yaml.safe_load(hosts_file.read_text()) or {}
        token = (hosts.get("github.com") or {}).get("oauth_token", "").strip()
        if token:
            return token

    raise RuntimeError(
        "No GitHub token found. Set the GITHUB_TOKEN environment variable "
        "or log in with `gh auth login`."
    )


def _call_copilot_api(prompt: str, model: str, temperature: float) -> str:
    """Call GitHub Copilot API and return raw response text."""
    from openai import OpenAI

    token = _get_github_token()
    client = OpenAI(base_url=_COPILOT_BASE_URL, api_key=token)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def _call_deepseek_api(prompt: str, model: str, temperature: float, api_key: str) -> str:
    """Call DeepSeek API and return raw response text."""
    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def _strip_markdown_fences(raw: str) -> str:
    """Remove accidental markdown code fences from LLM output."""
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.splitlines()[:-1])
    return raw


def generate_model_json(
    calc,
    model_name: str,
    h: int,
    config=None,
    llm_config: "LLMConfig" = None,
    write: bool = True,
    out_dir: "Path | str | None" = None,
) -> dict:
    """
    Call an LLM API (Copilot or DeepSeek) to generate / improve a ``model.json`` file.

    Args:
        calc:          A :class:`~nospy.features.FeaturesCalculator` instance
                       (features will be computed lazily if not yet done).
        model_name:    e.g. ``"nhits"``, ``"nbeats"``, ``"tft"``
                       (case-insensitive; leading ``"auto"`` stripped).
        h:             Forecast horizon.
        config:        Optional :class:`~nospy.config.ExperimentConfig` or
                       plain ``dict`` with ``num_samples`` / ``cpus`` / ``gpus``.
        llm_config:    Optional :class:`~nospy.config.LLMConfig` instance.
                       If ``None``, defaults to Copilot with ``gpt-4o``.
        write:         If ``True`` (default), replace ``json/<model>.json`` with
                       the generated config.
        out_dir:       Optional run output directory.  When supplied, two files
                       are written there:
                       - ``<out_dir>/<model>.json``          — generated config
                       - ``<out_dir>/<model>_features.json`` — feature summary

    Returns:
        Parsed ``dict`` with ``fixed``, ``run``, and ``test`` sections.
    """
    from nospy.config import LLMConfig

    if llm_config is None:
        raise ValueError("llm_config must be provided, not None")

    model_key = model_name.lower().replace("auto", "")
    json_path = _MODEL_CONFIG_DIR / f"{model_key}.json"

    existing_json: dict | None = None
    if json_path.exists():
        existing_json = json.loads(json_path.read_text())

    summary = calc.summarize()

    prompt = build_model_prompt(
        summary=summary,
        model_name=model_name,
        h=h,
        config=config,
        existing_json=existing_json,
    )

    if llm_config.provider == "deepseek":
        api_key = llm_config.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable "
                "or pass api_key in LLMConfig."
            )
        raw = _call_deepseek_api(
            prompt=prompt,
            model=llm_config.model or "deepseek-chat",
            temperature=llm_config.temperature,
            api_key=api_key,
        )
    else:
        # Default to Copilot
        raw = _call_copilot_api(
            prompt=prompt,
            model=llm_config.model or "gpt-4o",
            temperature=llm_config.temperature,
        )

    raw = _strip_markdown_fences(raw)
    new_cfg = json.loads(raw)

    # Enforce batch_size and windows_batch_size limits from original JSON
    if existing_json is not None:
        for param in ("batch_size", "windows_batch_size"):
            original_values = existing_json.get("run", {}).get(param, [])
            if original_values:
                max_original = max(original_values)
                new_values = new_cfg.get("run", {}).get(param, [])
                if new_values:
                    # Clamp each candidate to not exceed max_original
                    clamped = [min(v, max_original) for v in new_values]
                    # Remove duplicates while preserving order
                    seen = set()
                    unique_clamped = []
                    for v in clamped:
                        if v not in seen:
                            seen.add(v)
                            unique_clamped.append(v)
                    new_cfg["run"][param] = unique_clamped
                # Also clamp test value
                test_val = new_cfg.get("test", {}).get(param)
                if test_val is not None:
                    new_cfg["test"][param] = min(test_val, max_original)

    if write:
        json_path.write_text(json.dumps(new_cfg, indent=2))
        # Also save the prompt text to JSON directory
        prompt_path = _MODEL_CONFIG_DIR / f"{model_key}_prompt.txt"
        prompt_path.write_text(prompt)
        print(f"Prompt text saved to {prompt_path}")

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{model_key}.json").write_text(json.dumps(new_cfg, indent=2))
        (out / f"{model_key}_prompt.txt").write_text(prompt)

    return new_cfg
