# utils.py
import os
import pathlib
import tempfile
import time
from threading import Thread
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ---- basic config & env ----
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")  # set your hf_... token in env/Secrets
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # avoid flaky xet downloader


def pick_cache_dir() -> str:
    for p in (
        os.getenv("CACHE_DIR"),
        "/data/hf_cache" if os.path.isdir("/data") else None,  # HF Spaces / containers
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.join(tempfile.gettempdir(), "hf_cache"),
    ):
        if not p:
            continue
        try:
            pathlib.Path(p).mkdir(parents=True, exist_ok=True)
            t = pathlib.Path(p) / ".w"
            t.write_text("ok")
            t.unlink(missing_ok=True)
            return p
        except Exception:
            continue
    return os.path.join(tempfile.gettempdir(), "hf_cache")


CACHE_DIR = pick_cache_dir()
os.environ.setdefault("HF_HUB_CACHE", CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", CACHE_DIR)

# dtype/device
USE_CUDA = torch.cuda.is_available()
DTYPE = torch.float16 if USE_CUDA else torch.float32


class LlamaChatbot:
    """
    Llama-style CausalLM wrapper with throughput/latency/memory instrumentation.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        hf_token: Optional[str] = HF_TOKEN,
        cache_dir: str = CACHE_DIR,
        torch_dtype: torch.dtype = DTYPE,
        device_map: str = "auto",
        kv_dtype_bytes: int = 2,  # typical FP16 KV-cache; set to 1 if using int8 KV
    ) -> None:
        self.model_id = model_id
        self.hf_token = hf_token
        self.cache_dir = cache_dir
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.kv_dtype_bytes = kv_dtype_bytes

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.load_error: Optional[str] = None

        # Custom kernel patch bookkeeping
        self._patched_layers: int = 0
        self._patch_error: Optional[str] = None

    # ---------- lifecycle ----------
    def load(self) -> None:
        """Load tokenizer and model into memory. Safe to call multiple times."""
        try:
            self.load_error = None
            self._patch_error = None
            self._patched_layers = 0

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=self.hf_token,
                cache_dir=self.cache_dir,
                use_fast=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                token=self.hf_token,
                cache_dir=self.cache_dir,
                device_map=self.device_map,  # let Accelerate place layers
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
            )
            if (
                self.tokenizer is not None
                and self.tokenizer.pad_token_id is None
                and self.tokenizer.eos_token_id is not None
            ):
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.model is not None:
                self.model.eval()

            # ---- custom CUDA kernel patch (RMSNorm) ----
            try:
                if USE_CUDA and self.model is not None:
                    from custom_kernels import patch_llama_rmsnorm

                    n = patch_llama_rmsnorm(self.model)
                    self._patched_layers = int(n)
                    if n > 0:
                        print(f"[custom_kernels] Patched {n} LlamaRMSNorm layers with CUDA kernel.")
                    else:
                        print("[custom_kernels] No LlamaRMSNorm modules found to patch.")
            except Exception as e:
                self._patch_error = str(e)
                print(f"[custom_kernels] Patch failed, proceeding without: {e}")

        except Exception as e:
            self.load_error = str(e)
            self.tokenizer = None
            self.model = None

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None and self.load_error is None

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self.is_loaded() else "degraded",
            "model": self.model_id,
            "using_token": bool(self.hf_token),
            "cache_dir": self.cache_dir,
            "device": "cuda" if USE_CUDA else "cpu",
            "load_error": self.load_error,
            "custom_kernels": {
                "patched_layers": self._patched_layers,
                "patch_error": self._patch_error,
            },
        }

    # ---------- helpers ----------
    def _device(self):
        if self.model is None:
            return torch.device("cuda" if USE_CUDA else "cpu")
        return next(self.model.parameters()).device

    def _estimate_kv_cache_mb(self, seq_len: int) -> Optional[float]:
        """
        Estimate KV-cache size per sequence (batch item) in MiB.
        Uses model config: layers, kv heads, head_dim; assumes FP16 KV by default.
        """
        if self.model is None:
            return None
        cfg = self.model.config
        try:
            n_layers = int(getattr(cfg, "num_hidden_layers"))
            n_heads = int(getattr(cfg, "num_attention_heads"))
            n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
            head_dim = int(getattr(cfg, "hidden_size") // n_heads)
            bytes_per_val = int(self.kv_dtype_bytes)  # e.g., 2 for fp16, 1 for int8
            # K + V
            total_bytes = n_layers * n_kv * seq_len * head_dim * 2 * bytes_per_val
            return float(total_bytes) / (1024.0 * 1024.0)
        except Exception:
            return None

    # ---------- public API ----------
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Return a dict with generated_text + throughput, latency, memory stats.
        Measures TTFT via streaming. Guarantees the stream closes on error.
        """
        if not self.is_loaded():
            raise RuntimeError(f"Model not loaded: {self.load_error}")
        assert self.model is not None and self.tokenizer is not None

        # ---- tokenize & move ----
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False)
        device = self._device()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[1])

        # ---- prepare streaming for TTFT ----
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            streamer=streamer,
        )

        # ---- memory baseline & timers ----
        base_alloc = None
        if USE_CUDA:
            torch.cuda.synchronize(device)
            base_alloc = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)

        t0 = time.perf_counter()
        first_tok_time: Optional[float] = None
        out_holder: Dict[str, Any] = {}

        def _run():
            try:
                out_holder["out"] = self.model.generate(**gen_kwargs)
            except Exception as e:
                out_holder["error"] = str(e)
            finally:
                # Ensure the consumer iterator stops even on exceptions.
                try:
                    streamer.end()
                except Exception:
                    pass

        th = Thread(target=_run, daemon=True)
        th.start()

        # Stream to detect first token emission (TTFT)
        collected_chunks = []
        for chunk in streamer:
            if first_tok_time is None:
                first_tok_time = time.perf_counter()
            collected_chunks.append(chunk)

        th.join()
        t_end = time.perf_counter()
        if USE_CUDA:
            torch.cuda.synchronize(device)

        # If generation failed, surface the error
        if "error" in out_holder:
            raise RuntimeError(out_holder["error"])

        # ---- outputs & counts ----
        out = out_holder.get("out")
        if out is not None and hasattr(out, "sequences"):
            total_tokens = int(out.sequences.shape[1])
        else:
            total_tokens = prompt_tokens + len(
                self.tokenizer("".join(collected_chunks)).input_ids
            )

        generated_tokens = max(0, total_tokens - prompt_tokens)
        generated_text = "".join(collected_chunks)

        # ---- timings ----
        end_to_end = t_end - t0
        if first_tok_time is None:
            # no tokens emitted (e.g., max_new_tokens=0)
            first_tok_time = t_end
        ttft = first_tok_time - t0
        prefill_time = max(1e-9, ttft)
        decode_time = max(1e-9, t_end - first_tok_time)

        # ---- throughput ----
        prefill_tps = float(prompt_tokens) / prefill_time
        decode_tps = float(generated_tokens) / decode_time
        effective_tps = float(prompt_tokens + generated_tokens) / end_to_end

        # ---- memory stats ----
        peak_vram_mb = request_vram_delta_mb = None
        if USE_CUDA:
            peak_alloc = torch.cuda.max_memory_allocated(device)
            peak_vram_mb = peak_alloc / (1024.0 * 1024.0)
            if base_alloc is not None:
                request_vram_delta_mb = max(
                    0.0, (peak_alloc - base_alloc) / (1024.0 * 1024.0)
                )

        kv_cache_mb = self._estimate_kv_cache_mb(prompt_tokens + generated_tokens)

        return {
            "generated_text": generated_text,
            "throughput": {
                "prefill_tokens_per_s": prefill_tps,
                "decode_tokens_per_s": decode_tps,
                "effective_tokens_per_s": effective_tps,
            },
            "latency": {
                "ttft_s": ttft,
                "end_to_end_s": end_to_end,
                "prefill_s": prefill_time,
                "decode_s": decode_time,
            },
            "memory": {
                "peak_vram_mb": peak_vram_mb,
                "per_request_vram_mb": request_vram_delta_mb,  # renamed for clarity
                "kv_cache_size_per_seq_mb": kv_cache_mb,
            },
            "counts": {
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "total_tokens": prompt_tokens + generated_tokens,
            },
        }
# Optional: a ready-to-use singleton
chatbot = LlamaChatbot()
