import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open


@dataclass(frozen=True)
class Qwen3Cfg:
    # NOTE: Qwen3-4B config

    vocab_size: int = 151_936
    emb_dim: int = 2560
    n_transf_blocks: int = 36
    ffn_dim: int = 9728

    # Grouped-query attention (GQA)
    n_q_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128

    # Rotary position embeddings (RoPE)
    max_context_len: int = 40_960
    rope_base: float = 1_000_000.0

    rms_norm_eps: float = 1e-6

    dtype: torch.dtype = torch.float16

    @property
    def n_kv_groups(self) -> int:
        """Number of query heads that share each key/value head."""
        return self.n_q_heads // self.n_kv_heads

    @property
    def attn_inner_dim(self) -> int:
        """Width produced by the query projection."""
        return self.n_q_heads * self.head_dim

    @property
    def kv_inner_dim(self) -> int:
        """Width produced by each key and value projection."""
        return self.n_kv_heads * self.head_dim


cfg = Qwen3Cfg()

class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.normalized_shape = (emb_dim,)
        self.gamma = nn.Parameter(
            torch.ones(emb_dim, dtype=cfg.dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            x,
            normalized_shape=self.normalized_shape,
            weight=self.gamma,
            eps=cfg.rms_norm_eps,
        )

class Rope(nn.Module):
    """Qwen3 rotary position embeddings with precomputed sin/cos values."""

    # register_buffer assigns these dynamically; annotations keep type checkers
    # from treating them as generic nn.Module attributes.
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(
        self,
        max_seq_len: int,
        head_dim: int,
        base: float,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.head_dim = head_dim

        # One angular frequency for each pair of head dimensions.
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / head_dim
            )
        )  # [head_dim // 2]

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        )  # [max_seq_len]

        angles = positions[:, None] * inv_freq[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        # [max_seq_len, head_dim]

        # RoPE is deterministic, so these buffers do not belong in checkpoints.
        self.register_buffer(
            "cos",
            angles.cos(),
            persistent=False,
        )
        self.register_buffer(
            "sin",
            angles.sin(),
            persistent=False,
        )

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate Qwen's first and second head-dimension halves."""
        first_half, second_half = x.chunk(2, dim=-1)
        return torch.cat((-second_half, first_half), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Apply RoPE to [heads, sequence, head_dim] Q or K tensors."""
        if x.ndim != 3:
            raise ValueError(
                "RoPE expects [heads, sequence, head_dim], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"Expected head dimension {self.head_dim}, "
                f"got {x.shape[-1]}"
            )

        seq_len = x.shape[-2]
        end_pos = start_pos + seq_len
        if start_pos < 0 or end_pos > self.max_seq_len:
            raise ValueError(
                f"Positions [{start_pos}, {end_pos}) exceed the "
                f"precomputed range [0, {self.max_seq_len})"
            )

        # [sequence, head_dim] broadcasts over the head dimension.
        cos = self.cos[start_pos:end_pos].to(dtype=x.dtype)
        sin = self.sin[start_pos:end_pos].to(dtype=x.dtype)

        return (x * cos) + (self.rotate_half(x) * sin)

class GQA(nn.Module):
    causal_mask: torch.Tensor
    def __init__(self):
        super().__init__()

        # 2560 -> 4096
        self.Wq = nn.Linear(
            cfg.emb_dim,
            cfg.n_q_heads * cfg.head_dim,
            dtype=cfg.dtype,
            bias=False,
        )

        # 2560 -> 1024
        self.Wk = nn.Linear(
            cfg.emb_dim,
            cfg.kv_inner_dim,
            dtype=cfg.dtype,
            bias=False,
        )

        # 2560 -> 1024
        self.Wv = nn.Linear(
            cfg.emb_dim,
            cfg.kv_inner_dim,
            dtype=cfg.dtype,
            bias=False,
        )

        # 4096 -> 2560: Back to original embedding dimension
        self.Wo = nn.Linear(
            cfg.n_q_heads * cfg.head_dim,
            cfg.emb_dim,
            dtype=cfg.dtype,
            bias=False,
        )

        # Norm over head dim: 128
        self.qnorm = RMSNorm(cfg.head_dim)
        self.knorm = RMSNorm(cfg.head_dim)

        self.scale = cfg.head_dim ** -0.5
        self.rope = Rope(
            max_seq_len=cfg.max_context_len,
            head_dim=cfg.head_dim,
            base=cfg.rope_base,
        )

        # Grow this non-persistent cache on demand instead of allocating a
        # 40,960 x 40,960 mask in advance (~1.68 GB).
        self.register_buffer(
            "causal_mask",
            torch.empty(0, 0, dtype=torch.bool),
            persistent=False,
        )

    def get_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if (
            self.causal_mask.device != device
            or self.causal_mask.shape[0] < seq_len
        ):
            self.causal_mask = torch.triu(
                torch.ones(
                    seq_len,
                    seq_len,
                    dtype=torch.bool,
                    device=device,
                ),
                diagonal=1,
            )

        return self.causal_mask[:seq_len, :seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = [sequence, emb_dim]
        seq_len = x.shape[0]

        q = self.Wq(x) # [seq_len, 4096]
        k = self.Wk(x) # [seq_len, 1024]
        v = self.Wv(x) # [seq_len, 1024]

        # split into heads: 4096 -> [32, 128] | 1024 -> [8, 128]
        q = q.view(seq_len, cfg.n_q_heads, cfg.head_dim)
        k = k.view(seq_len, cfg.n_kv_heads, cfg.head_dim)
        v = v.view(seq_len, cfg.n_kv_heads, cfg.head_dim)

        #XXX: Apply q, k norm
        q = self.qnorm(q)
        k = self.knorm(k)

        #XXX: Compute q, k similarity scores
        # want to multiply the first 4 heads of q with the first head of k.T
        # single head looks like [seq_len, head_dim: 128]

        # move the heads dim to the start
        q = q.transpose(0, 1) # [n_q_heads: 32, seq_len, head_dim: 128]
        k = k.transpose(0, 1) # [8, seq_len, 128]
        v = v.transpose(0, 1) # [8, seq_len, 128]

        # Apply position-dependent rotations to Q and K only.
        q = self.rope(q)
        k = self.rope(k)

        # [8, 4, seq_len, 128]
        q_4_group = q.reshape(cfg.n_kv_heads, cfg.n_kv_groups, seq_len, cfg.head_dim)

        # [8, 1, seq_len, 128]
        #     ^ will get broadcasted (duplicated 4 times) in multiply below
        k_1_group = k.unsqueeze(1)
        k_1_group_T = k_1_group.transpose(-2, -1) # swap seq_len, 128 -> 128, seq_len

        # [seq_len, 128] @ [128, seq_len] = [seq_len, seq_len] similarity scores
        scores = q_4_group @ k_1_group_T # [8, 4, seq_len, seq_len]

        # Scale the logits before masking and softmax.
        scores = scores * self.scale

        # Reuse the lazy mask cache registered in __init__.
        causal_mask = self.get_causal_mask(seq_len, scores.device)

        # The mask broadcasts over [n_kv_heads, n_kv_groups].
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Normalize each query row over its allowed key positions.
        attn_scores_norm = F.softmax(
            scores,
            dim=-1,
            dtype=torch.float32,
        ).to(scores.dtype)

        #XXX: Compute attention output: GEMM V, re-concat 32x128 heads into 4096 dim, multiply by Wo so 4096 -> 2560
        v_1_group = v.unsqueeze(1) # [8, 1, seq_len, 128]

        # multiply all attention scores with their v values
        attn_output = attn_scores_norm @ v_1_group # [8, 4, seq_len, 128]

        attn_output = attn_output.reshape(cfg.n_q_heads, seq_len, cfg.head_dim) # [32, seq_len, 128]

        # move seq before heads so we can rebuild the 4096 dimension representation per token
        attn_output = attn_output.transpose(0, 1) # [seq_len, 32, 128]

        # concatenate the 32 heads -> [seq_len, 32x128=4096]
        attn_output = attn_output.reshape(seq_len, cfg.n_q_heads * cfg.head_dim)

        out = self.Wo(attn_output) #4096 -> 2560

        return out


class FFN(nn.Module):
    def __init__(self):
        super().__init__()

        self.Lin1 = nn.Linear(
            cfg.emb_dim,
            cfg.ffn_dim,
            bias=False,
            dtype=cfg.dtype,
        )
        self.Lin2 = nn.Linear(
            cfg.emb_dim,
            cfg.ffn_dim,
            bias=False,
            dtype=cfg.dtype,
        )
        self.Lin3 = nn.Linear(
            cfg.ffn_dim,
            cfg.emb_dim,
            bias=False,
            dtype=cfg.dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_lin1 =  self.Lin1(x)
        x_lin2 = self.Lin2(x)
        x = nn.functional.silu(x_lin1) * x_lin2
        return self.Lin3(x)

class TransformerBlk(nn.Module):
    def __init__(self):
        super().__init__()
        self.gqa = GQA()
        self.ffn = FFN()
        self.rmsnorm1 = RMSNorm(cfg.emb_dim)
        self.rmsnorm2 = RMSNorm(cfg.emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.rmsnorm1(x)
        x = self.gqa(x)
        x += shortcut

        shortcut2 = x

        x = self.rmsnorm2(x)
        x = self.ffn(x)

        x += shortcut2

        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_tokens = nn.Embedding(
            cfg.vocab_size,
            cfg.emb_dim,
            dtype=cfg.dtype,
        )
        self.trf_blks = nn.ModuleList(
            [TransformerBlk() for _ in range(cfg.n_transf_blocks)]
        )
        self.rmsnorm_final = RMSNorm(cfg.emb_dim)
        self.lm_head = nn.Linear(
            cfg.emb_dim,
            cfg.vocab_size,
            bias=False,
            dtype=cfg.dtype,
        )

        # Qwen3-4B shares the input embedding and output projection weights.
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # This implementation accepts one unbatched sequence: [seq_len].
        if input_ids.ndim != 1:
            raise ValueError(
                "Expected input_ids with shape [seq_len], "
                f"got {tuple(input_ids.shape)}"
            )
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must have dtype torch.long")
        if input_ids.numel() == 0:
            raise ValueError("input_ids cannot be empty")
        if input_ids.shape[0] > cfg.max_context_len:
            raise ValueError(
                f"Sequence length {input_ids.shape[0]} exceeds "
                f"maximum context length {cfg.max_context_len}"
            )

        # Token IDs -> hidden vectors: [seq_len] -> [seq_len, 2560].
        x = self.embed_tokens(input_ids)

        for block in self.trf_blks:
            x = block(x)

        x = self.rmsnorm_final(x)

        # [seq_len, 2560] -> [seq_len, 151936].
        logits = self.lm_head(x)
        return logits

### CHECKPOINT LOADING


def hf_to_local_parameter_name(hf_name: str) -> str:
    """Translate a Qwen3 Hugging Face parameter name to this model's name."""
    if hf_name == "model.embed_tokens.weight":
        return "embed_tokens.weight"
    if hf_name == "model.norm.weight":
        return "rmsnorm_final.gamma"

    prefix = "model.layers."
    if not hf_name.startswith(prefix):
        raise KeyError(f"Unsupported Hugging Face parameter: {hf_name}")

    remainder = hf_name[len(prefix):]
    layer_text, separator, layer_parameter = remainder.partition(".")
    if not separator or not layer_text.isdigit():
        raise KeyError(f"Malformed Hugging Face parameter: {hf_name}")

    layer = int(layer_text)
    layer_prefix = f"trf_blks.{layer}."
    layer_parameter_map = {
        "input_layernorm.weight": "rmsnorm1.gamma",
        "post_attention_layernorm.weight": "rmsnorm2.gamma",
        "self_attn.q_proj.weight": "gqa.Wq.weight",
        "self_attn.k_proj.weight": "gqa.Wk.weight",
        "self_attn.v_proj.weight": "gqa.Wv.weight",
        "self_attn.o_proj.weight": "gqa.Wo.weight",
        "self_attn.q_norm.weight": "gqa.qnorm.gamma",
        "self_attn.k_norm.weight": "gqa.knorm.gamma",
        "mlp.gate_proj.weight": "ffn.Lin1.weight",
        "mlp.up_proj.weight": "ffn.Lin2.weight",
        "mlp.down_proj.weight": "ffn.Lin3.weight",
    }

    try:
        return layer_prefix + layer_parameter_map[layer_parameter]
    except KeyError as error:
        raise KeyError(
            f"Unsupported layer parameter: {hf_name}"
        ) from error


@torch.no_grad()
def load_qwen_checkpoint(
    model: Model,
    repo_id: str = "Qwen/Qwen3-4B",
 ) -> Path:
    """Download and stream the sharded BF16 checkpoint into this FP16 model."""
    checkpoint_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[
                "model-*.safetensors",
                "model.safetensors.index.json",
            ],
        )
    )

    index_path = checkpoint_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as index_file:
        weight_map: dict[str, str] = json.load(index_file)["weight_map"]

    model_parameters = dict(model.named_parameters())
    loaded_parameters: set[str] = set()

    shard_to_weights: dict[str, list[str]] = {}
    for hf_name, shard_name in weight_map.items():
        shard_to_weights.setdefault(shard_name, []).append(hf_name)

    for shard_number, shard_name in enumerate(sorted(shard_to_weights), start=1):
        print(
            f"Loading checkpoint shard {shard_number}/"
            f"{len(shard_to_weights)}: {shard_name}"
        )

        shard_path = checkpoint_dir / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            for hf_name in shard_to_weights[shard_name]:
                local_name = hf_to_local_parameter_name(hf_name)

                if local_name not in model_parameters:
                    raise KeyError(
                        f"Checkpoint parameter {hf_name} maps to missing "
                        f"model parameter {local_name}"
                    )

                source = shard.get_tensor(hf_name)
                destination = model_parameters[local_name]

                if source.shape != destination.shape:
                    raise ValueError(
                        f"Shape mismatch for {hf_name} -> {local_name}: "
                        f"checkpoint={tuple(source.shape)}, "
                        f"model={tuple(destination.shape)}"
                    )

                # Convert BF16 to FP16 on CPU before transferring to a P100 or MPS.
                source = source.to(dtype=destination.dtype)
                destination.copy_(source.to(device=destination.device))
                loaded_parameters.add(local_name)

    missing = set(model_parameters) - loaded_parameters
    if missing:
        raise RuntimeError(
            "The checkpoint did not initialize these parameters:\n"
            + "\n".join(sorted(missing))
        )

    # Restore the shared Parameter in case model construction changes later.
    model.lm_head.weight = model.embed_tokens.weight
    return checkpoint_dir


### GENERATION

from transformers import AutoTokenizer, TextStreamer


MODEL_ID = "Qwen/Qwen3-4B"


def load_tokenizer():
    """Load Qwen's tokenizer and chat template from Hugging Face."""
    return AutoTokenizer.from_pretrained(MODEL_ID)


@torch.inference_mode()
def generate_greedy(
    model: Model,
    input_ids: torch.Tensor,
    stop_token_ids: set[int],
    max_new_tokens: int = 256,
    streamer=None,
) -> list[int]:
    """Generate one token at a time by recomputing the full prefix."""
    if input_ids.ndim != 1:
        raise ValueError(
            "Expected input_ids with shape [seq_len], "
            f"got {tuple(input_ids.shape)}"
        )

    generated_token_ids: list[int] = []

    for _ in range(max_new_tokens):
        if input_ids.shape[0] >= cfg.max_context_len:
            break

        # [seq_len, vocab_size]; only the last row predicts the next token.
        logits = model(input_ids)
        next_token = torch.argmax(logits[-1].float(), dim=-1)
        next_token_id = int(next_token.item())

        if next_token_id in stop_token_ids:
            break

        generated_token_ids.append(next_token_id)
        input_ids = torch.cat((input_ids, next_token.view(1)))

        if streamer is not None:
            streamer.put(next_token.detach().cpu().view(1))

    if streamer is not None:
        streamer.end()

    return generated_token_ids


def chat_loop(
    model: Model,
    tokenizer,
    max_new_tokens: int = 256,
 ) -> None:
    """Run a single-user, unbatched terminal chat session."""
    device = next(model.parameters()).device
    model.eval()

    stop_token_ids = {
        token_id
        for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id)
        if token_id is not None
    }

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
    ]

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})

        # apply_chat_template returns [batch=1, seq_len]; remove the batch axis.
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )[0].to(device)

        print("Assistant: ", end="", flush=True)
        streamer = TextStreamer(
            tokenizer,
            skip_prompt=False,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        generated_token_ids = generate_greedy(
            model=model,
            input_ids=input_ids,
            stop_token_ids=stop_token_ids,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
        )

        # Decode again so the exact assistant response can be saved in history.
        assistant_text = tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        messages.append(
            {"role": "assistant", "content": assistant_text}
        )


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        raise RuntimeError(
            "Qwen3-4B requires a CUDA or Apple MPS accelerator"
        )

    print(f"Using device: {device}")
    print("Constructing the FP16 model...")

    # Construct directly on the accelerator to avoid a second model-sized copy.
    with torch.device(device):
        model = Model()

    model.eval()

    print("Loading Qwen3-4B checkpoint...")
    load_qwen_checkpoint(model)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("Model ready. Type 'exit' or 'quit' to stop.")
    chat_loop(model, tokenizer)


if __name__ == "__main__":
    main()

