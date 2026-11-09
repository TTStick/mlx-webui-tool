"""测试夹具：现场生成一个随机权重的微型 Llama 模型。

它是真实的 MLX 模型，走完整的 load / stream_generate / KV 缓存流程，
只是输出内容是随机的。这样测试不需要联网下载任何东西。
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MLXUI_DATA_DIR", tempfile.mkdtemp(prefix="mlxui-test-"))

import mlx.core as mx  # noqa: E402
import pytest  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers  # noqa: E402

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


# 与 Qwen3 的行为一致：开启时模板以 <think> 结尾，关闭时写入一段空的思考块
THINK_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n"
    "{% else %}<think>\n{% endif %}{% endif %}"
)


class MemoryKeyStore:
    """代替钥匙串，测试不碰用户真实的钥匙串。"""

    def __init__(self) -> None:
        self.key: str | None = None

    def get(self) -> str | None:
        return self.key

    def put(self, key: str) -> None:
        self.key = key

    def delete(self) -> None:
        self.key = None


def make_tiny_model(path: Path, thinking: bool = False) -> None:
    from mlx_lm.models.llama import Model, ModelArgs

    path.mkdir(parents=True, exist_ok=True)
    charset = [
        chr(c)
        for block in ((32, 127), (0x2000, 0x2070), (0x3000, 0x3040), (0x4E00, 0x9FA6), (0xFF00, 0xFF60))
        for c in range(*block)
    ]
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<unk>", "\n"]
    if thinking:
        specials += ["<think>", "</think>"]
    vocab = {token: i for i, token in enumerate(specials + charset)}

    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([])
    tokenizer.decoder = decoders.Fuse()
    tokenizer.add_special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<unk>"])
    if thinking:
        tokenizer.add_tokens(["<think>", "</think>"])
    # WordLevel 按整词匹配；用正则预分词把文本切成单字符
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), "isolated")
    tokenizer.save(str(path / "tokenizer.json"))

    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": "<|im_end|>",
                "pad_token": "<|endoftext|>",
                "unk_token": "<unk>",
                "chat_template": THINK_TEMPLATE if thinking else CHAT_TEMPLATE,
                "model_max_length": 4096,
            }
        )
    )
    config = {
        "model_type": "llama",
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-5,
        "vocab_size": len(vocab),
        "max_position_embeddings": 4096,
        "eos_token_id": vocab["<|im_end|>"],
        "bos_token_id": None,
        "tie_word_embeddings": False,
    }
    (path / "config.json").write_text(json.dumps(config))
    args = ModelArgs.from_dict(config)
    mx.random.seed(1234)  # 固定权重，让贪心采样的结果在每次运行中一致
    model = Model(args)
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(path / "model.safetensors"), weights, metadata={"format": "mlx"})


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("models")
    make_tiny_model(root / "tiny-llama")
    make_tiny_model(root / "tiny-think", thinking=True)
    return root


@pytest.fixture()
def app_ctx(tiny_model_dir, tmp_path):
    from fastapi.testclient import TestClient

    from mlxui.app import create_app

    app = create_app(allow_any_host=True, db_path=tmp_path / "test.db", keystore=MemoryKeyStore())
    app.state.db.update_settings(
        {
            "model_dirs": [str(tiny_model_dir)],
            "auto_title": False,
            "memory": {"enabled": False},  # 反思用到的调度由专门的测试覆盖
            "privacy": {"usage_stats": True},
        }
    )
    app.state.registry.invalidate()
    with TestClient(app) as client:
        models = client.get("/api/models").json()["models"]
        assert models, "fixture model was not discovered"
        yield client, app, models[0]["id"]


def wait_job(app, job_id: str, timeout: float = 90.0):
    job = app.state.engine.jobs[job_id]
    deadline = time.time() + timeout
    while not job.finished:
        assert time.time() < deadline, "job timed out"
        time.sleep(0.02)
    return job
