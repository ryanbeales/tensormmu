"""Contrastive dataset construction for Neural VM calibration and evaluation.

Step 4.1 Specification:
- Construct two contrastive calibration sets of 128 sequences each (seq_len = 256):
  1. Logic_Set (X_logic): Pure symbolic reasoning, Python control flow/algorithms,
     and multi-step arithmetic (sourced from gsm8k or synthetic algorithmic traces).
     Minimize named entities or historical trivia.
  2. Fact_Set (X_fact): High-density encyclopedic trivia, obscure entity relations,
     dates, and API lookup strings (sourced from wikitext or trivia_qa).
- Also produces held-out Logic_Eval and Fact_Eval tensors for ablation benchmarking.
"""

from dataclasses import dataclass
import random
import re
from typing import Any, List, Optional, Tuple
import torch


@dataclass
class TokenizedSplit:
    """Batch-ready tokenized dataset split of fixed shape [num_seqs, seq_len]."""

    name: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    @property
    def num_sequences(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])


@dataclass
class ContrastiveCalibrationBundle:
    """Holds calibration, validation, and evaluation splits for Logic and Fact domains."""

    logic_calib: TokenizedSplit
    fact_calib: TokenizedSplit
    logic_val: TokenizedSplit
    fact_val: TokenizedSplit
    logic_eval: TokenizedSplit
    fact_eval: TokenizedSplit


_GENERIC_VARS = ["x", "y", "z", "n", "k", "m", "acc", "delta", "idx", "val", "state", "ptr"]


def _clean_wikitext_line(text: str) -> str:
    """Detokenizes Moses/Penn-Treebank artifacts in WikiText (e.g. ' @-@ ', ' ,', ' .')."""
    text = text.replace(" @-@ ", "-").replace(" @.@ ", ".").replace(" @,@ ", ",")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+('s|'re|'ve|'d|'ll|n't)\b", r"\1", text)
    text = text.replace("( ", "(").replace(" )", ")").replace("[ ", "[").replace(" ]", "]")
    text = re.sub(r'"\s+([^"]*?)\s+"', r'"\1"', text)
    return text


def _anonymize_named_entities(text: str) -> str:
    """Replaces capitalized proper names in GSM8K problems with symbolic identifiers."""
    words = text.split()
    mapping = {}
    var_pool = ["A", "B", "C", "D", "E", "F", "X", "Y", "Z"]
    out = []
    for i, w in enumerate(words):
        stripped = re.sub(r"[^A-Za-z]", "", w)
        # Replace mid-sentence capitalized words that look like proper names
        if (
            i > 0
            and stripped
            and stripped[0].isupper()
            and stripped[1:].islower()
            and stripped.lower()
            not in {"if", "then", "else", "while", "for", "return", "step", "total", "calculate", "find", "given"}
        ):
            if stripped not in mapping:
                mapping[stripped] = f"agent_{var_pool[len(mapping) % len(var_pool)]}"
            out.append(w.replace(stripped, mapping[stripped]))
        else:
            out.append(w)
    return " ".join(out)


def _generate_synthetic_logic_trace(rng: random.Random, idx: int) -> str:
    """Generates pure symbolic reasoning, Python control flow, and multi-step arithmetic."""
    a = rng.randint(11, 97)
    b = rng.randint(3, 29)
    c = rng.randint(5, 43)
    d = rng.randint(2, 13)

    step1 = a * b + c
    step2 = (step1 - d) * 2
    step3 = step2 // d
    rem = step2 % d

    v1, v2, v3 = rng.sample(_GENERIC_VARS, 3)

    code_block = (
        f"def compute_state_transition({v1}: int, {v2}: int, {v3}: int) -> int:\n"
        f"    acc = 0\n"
        f"    state = {v1}\n"
        f"    for i in range(1, {d} + 1):\n"
        f"        if (state + i) % 2 == 0:\n"
        f"            acc += (state * {v2}) - i\n"
        f"            state = (state // 2) + {v3}\n"
        f"        else:\n"
        f"            acc -= (state + {v3})\n"
        f"            state = state * 3 + 1\n"
        f"    return acc ^ state\n"
    )

    proof_block = (
        f"Symbolic Proof Trace #{idx}:\n"
        f"Given integer invariants x_0 = {a}, y_0 = {b}, z_0 = {c}, modulus m = {d}.\n"
        f"1. Evaluate linear combination: s_1 = x_0 * y_0 + z_0 = {a} * {b} + {c} = {step1}.\n"
        f"2. Apply affine shift and scaling: s_2 = 2 * (s_1 - m) = 2 * ({step1} - {d}) = {step2}.\n"
        f"3. Euclidean division by m = {d}: quotient q = {step3}, remainder r = {rem} "
        f"since {d} * {step3} + {rem} == {step2}.\n"
        f"4. Verify boolean invariant: (q > r) is {step3 > rem} and ((q + r) % 2 == 0) is {(step3 + rem) % 2 == 0}.\n"
        f"{code_block}"
        f"Invariant check: forall k >= 0, transition(x_k, y_k, z_k) preserves parity iff m is odd.\n"
    )
    return proof_block


def _generate_synthetic_fact_passage(rng: random.Random, idx: int) -> str:
    """Generates high-density encyclopedic trivia, obscure entity relations, dates, and API strings."""
    dynasties = [
        ("Ptolemaic Kingdom", "Alexandria", "305 BCE", "Ptolemy I Soter", "Lagid Hellenism"),
        ("Seleucid Empire", "Antioch-on-the-Orontes", "312 BCE", "Seleucus I Nicator", "Babylonian Satrapy"),
        ("Maurya Empire", "Pataliputra", "322 BCE", "Chandragupta Maurya", "Magadha Administration"),
        ("Achaemenid Empire", "Persepolis", "550 BCE", "Cyrus the Great", "Behistun Inscription"),
        ("Sasanian Empire", "Ctesiphon", "224 CE", "Ardashir I", "Eranshahr Chancery"),
        ("Axumite Kingdom", "Aksum", "100 CE", "Ezana of Axum", "Red Sea Trade Directorate"),
        ("Kushan Empire", "Bagram", "30 CE", "Kujula Kadphises", "Bactrian Epigraphy"),
        ("Khwarazmian Dynasty", "Gurganj", "1077 CE", "Anushtegin Gharchai", "Transoxiana Registry"),
    ]
    elements = [
        ("Ytterbium", "Yb", 70, "173.045 u", "Carl Gustaf Mosander", "Ytterby Quarry, Sweden", 1878),
        ("Praseodymium", "Pr", 59, "140.90766 u", "Carl Auer von Welsbach", "Vienna Imperial Lab", 1885),
        ("Gadolinium", "Gd", 64, "157.25 u", "Jean Charles Galissard de Marignac", "Geneva Spectroscopy", 1880),
        ("Thulium", "Tm", 69, "168.93422 u", "Per Teodor Cleve", "Uppsala University", 1879),
        ("Lutetium", "Lu", 71, "174.9668 u", "Georges Urbain", "Sorbonne Paris", 1907),
        ("Dysprosium", "Dy", 66, "162.500 u", "Paul Emile Lecoq de Boisbaudran", "Paris Academy", 1886),
    ]
    apis = [
        "GET /v2/catalog/entities/wikidata:Q184729/relations?include=chronology,treaties&format=application/vnd.geo+json",
        "POST /api/v3/taxonomy/ncbi:txid9606/orthologs?AssemblyAccession=GCF_000001405.40&refseq=NM_001126112.3",
        "GET /v1/astronomy/ephemeris/jpl:HORIZONS:2000433?epoch=2451545.0&ref_frame=ICRF_J2000&aberration=stellar",
        "PUT /v4/iso-standards/iec:61850-7-4:2010/substation/LNClass/MMXU/DataObject/TotW",
        "GET /v2/numismatics/ric/volume-iv/antoninus-pius/denarius:RIC_III_175a?mint=Rome&metal=AR",
    ]
    treaties = [
        ("Treaty of Nymphaeum", "13 March 1261", "Michael VIII Palaiologos", "Republic of Genoa"),
        ("Peace of Callias", "449 BCE", "Artaxerxes I of Persia", "Delian League Envoy Callias"),
        ("Treaty of Devol", "September 1108", "Bohemond I of Antioch", "Alexios I Komnenos"),
        ("Concordat of Worms", "23 September 1122", "Pope Callixtus II", "Holy Roman Emperor Henry V"),
        ("Treaty of Breda", "31 July 1667", "Charles II of England", "States General of the Dutch Republic"),
    ]

    dyn = dynasties[(idx + rng.randint(0, 7)) % len(dynasties)]
    elem = elements[(idx + rng.randint(0, 5)) % len(elements)]
    api = apis[(idx + rng.randint(0, 4)) % len(apis)]
    treaty = treaties[(idx + rng.randint(0, 4)) % len(treaties)]
    year = 1450 + (idx * 17) % 480
    lat = round(12.3456 + (idx % 60) * 0.713, 4)
    lon = round(24.8912 + (idx % 90) * 0.841, 4)

    return (
        f"Encyclopedic Compendium Entry #{idx}: The {dyn[0]} maintained its primary administrative capital at "
        f"{dyn[1]} (established c. {dyn[2]} under {dyn[3]}, archive classification: {dyn[4]}). "
        f"Diplomatic protocols later referenced the {treaty[0]}, ratified on {treaty[1]} between "
        f"{treaty[2]} and {treaty[3]} at coordinates {lat}N, {lon}E.\n"
        f"Mineralogical Assay ({year}): Lanthanide series specimen {elem[0]} (symbol {elem[1]}, atomic number "
        f"Z={elem[2]}, standard atomic weight {elem[3]}) was isolated by {elem[4]} at {elem[5]} in {elem[6]}.\n"
        f"Registry Endpoint: `{api}` | Checksum: SHA256-0x{idx * 2654435761 & 0xFFFFFFFF:08x}.\n"
    )


def _load_hf_dataset_cached_first(path: str, name: str, split: str):
    """Loads a Hugging Face dataset using local cache first if available."""
    import os

    cache_root = os.path.expanduser("~/.cache/huggingface/datasets")
    use_offline = os.path.isdir(cache_root)
    if use_offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    from datasets import load_dataset
    import datasets.config
    import huggingface_hub.constants

    if use_offline:
        datasets.config.HF_DATASETS_OFFLINE = True
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        try:
            return load_dataset(path, name, split=split)
        except Exception:
            pass

    os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ.pop("HF_HUB_OFFLINE", None)
    datasets.config.HF_DATASETS_OFFLINE = False
    huggingface_hub.constants.HF_HUB_OFFLINE = False
    return load_dataset(path, name, split=split)


def generate_logic_texts(total_needed: int, seed: int = 42, allow_hf_download: bool = True) -> List[str]:
    """Loads symbolic/arithmetic reasoning sequences from GSM8K + algorithmic traces."""
    rng = random.Random(seed)
    texts: List[str] = []

    if allow_hf_download:
        try:
            ds = _load_hf_dataset_cached_first("openai/gsm8k", "main", split="train")
            for row in ds:
                if len(texts) >= total_needed:
                    break
                q = _anonymize_named_entities(str(row.get("question", "")))
                a = _anonymize_named_entities(str(row.get("answer", "")))
                texts.append(
                    f"<start_of_turn>user\nSolve this symbolic reasoning problem step by step:\n{q}<end_of_turn>\n"
                    f"<start_of_turn>model\n{a}<end_of_turn>\n"
                )
        except Exception:
            pass

    while len(texts) < total_needed:
        idx = len(texts)
        block = "\n".join(_generate_synthetic_logic_trace(rng, idx * 3 + k) for k in range(3))
        texts.append(block)

    return texts[:total_needed]


def generate_fact_texts(total_needed: int, seed: int = 42, allow_hf_download: bool = True) -> List[str]:
    """Loads high-density encyclopedic trivia sequences from WikiText + factual lookup strings."""
    rng = random.Random(seed + 1000)
    texts: List[str] = []

    if allow_hf_download:
        try:
            ds = _load_hf_dataset_cached_first("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            buffer = []
            for row in ds:
                line = _clean_wikitext_line(str(row.get("text", "")).strip())
                if len(line) > 60 and not line.startswith("="):
                    buffer.append(line)
                if len(buffer) >= 2:
                    wiki_chunk = "\n".join(buffer)
                    buffer = []
                    topic = " ".join(wiki_chunk.split()[:6])
                    texts.append(
                        f"<start_of_turn>user\nProvide detailed encyclopedic facts about {topic}:<end_of_turn>\n"
                        f"<start_of_turn>model\n{wiki_chunk}<end_of_turn>\n"
                    )
                    if len(texts) >= total_needed:
                        break
        except Exception:
            pass

    while len(texts) < total_needed:
        idx = len(texts)
        block = "\n".join(_generate_synthetic_fact_passage(rng, idx * 3 + k) for k in range(3))
        texts.append(block)

    return texts[:total_needed]


def _tokenize_into_fixed_blocks(
    texts: List[str],
    tokenizer: Any,
    num_seqs: int,
    seq_len: int,
    split_name: str,
) -> TokenizedSplit:
    """Packs and tokenizes text passages into exact [num_seqs, seq_len] token blocks."""
    if tokenizer is None:
        raise ValueError("Tokenizer must not be None when tokenizing sequences.")

    bos_id = getattr(tokenizer, "bos_token_id", None)
    effective_len = seq_len - 1 if bos_id is not None else seq_len

    all_token_ids: List[int] = []
    target_total_tokens = num_seqs * effective_len

    for text in texts:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        ids = encoded["input_ids"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        all_token_ids.extend(ids)
        if len(all_token_ids) >= target_total_tokens:
            break

    # Cycle tokens if corpus was shorter than target_total_tokens
    if not all_token_ids:
        all_token_ids = [1] * target_total_tokens
    while len(all_token_ids) < target_total_tokens:
        all_token_ids.extend(all_token_ids[: target_total_tokens - len(all_token_ids)])

    trimmed = all_token_ids[:target_total_tokens]
    input_ids = torch.tensor(trimmed, dtype=torch.long).view(num_seqs, effective_len)
    if bos_id is not None:
        bos_col = torch.full((num_seqs, 1), int(bos_id), dtype=torch.long)
        input_ids = torch.cat([bos_col, input_ids], dim=1)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    return TokenizedSplit(
        name=split_name,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )


def build_contrastive_datasets(
    tokenizer: Any,
    num_calibration_seqs: int = 128,
    num_val_seqs: int = 16,
    num_eval_seqs: int = 32,
    seq_len: int = 256,
    seed: int = 42,
    allow_hf_download: bool = True,
) -> ContrastiveCalibrationBundle:
    """Builds Logic/Fact calibration (train), validation (dev), and held-out evaluation splits."""
    total_seqs = num_calibration_seqs + num_val_seqs + num_eval_seqs

    logic_texts = generate_logic_texts(total_needed=total_seqs * 2, seed=seed, allow_hf_download=allow_hf_download)
    fact_texts = generate_fact_texts(total_needed=total_seqs * 2, seed=seed, allow_hf_download=allow_hf_download)

    c_cut = num_calibration_seqs * 2
    v_cut = (num_calibration_seqs + num_val_seqs) * 2

    logic_calib = _tokenize_into_fixed_blocks(
        logic_texts[:c_cut],
        tokenizer=tokenizer,
        num_seqs=num_calibration_seqs,
        seq_len=seq_len,
        split_name="Logic_Set",
    )
    logic_val = _tokenize_into_fixed_blocks(
        logic_texts[c_cut:v_cut],
        tokenizer=tokenizer,
        num_seqs=num_val_seqs,
        seq_len=seq_len,
        split_name="Logic_Val",
    )
    logic_eval = _tokenize_into_fixed_blocks(
        logic_texts[v_cut:],
        tokenizer=tokenizer,
        num_seqs=num_eval_seqs,
        seq_len=seq_len,
        split_name="Logic_Eval",
    )

    fact_calib = _tokenize_into_fixed_blocks(
        fact_texts[:c_cut],
        tokenizer=tokenizer,
        num_seqs=num_calibration_seqs,
        seq_len=seq_len,
        split_name="Fact_Set",
    )
    fact_val = _tokenize_into_fixed_blocks(
        fact_texts[c_cut:v_cut],
        tokenizer=tokenizer,
        num_seqs=num_val_seqs,
        seq_len=seq_len,
        split_name="Fact_Val",
    )
    fact_eval = _tokenize_into_fixed_blocks(
        fact_texts[v_cut:],
        tokenizer=tokenizer,
        num_seqs=num_eval_seqs,
        seq_len=seq_len,
        split_name="Fact_Eval",
    )

    return ContrastiveCalibrationBundle(
        logic_calib=logic_calib,
        fact_calib=fact_calib,
        logic_val=logic_val,
        fact_val=fact_val,
        logic_eval=logic_eval,
        fact_eval=fact_eval,
    )
