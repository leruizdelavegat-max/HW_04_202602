"""Baselines (mediana, TF-IDF + Ridge, embeddings + Ridge) y BERT como regresor."""
import re
import time
from pathlib import Path

import numpy as np

from . import config

# ── Baseline 0: mediana ─────────────────────────────────────────────────────────


class Mediana:
    def fit(self, X, y):
        self.valor_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.valor_)


# ── Baseline 1: TF-IDF + Ridge ──────────────────────────────────────────────────

def _sin_digitos(t):
    return re.sub(r"\d", "#", t.lower())


def tfidf_ridge(alpha=1.0, chars=False, digitos=True, min_df=2):
    """Palabras 1–2 (+ opcional caracteres 3–5). `digitos=False` convierte cada dígito en '#'
    (el modelo ya no ve QUÉ año es, solo que hay uno): sirve para la Pregunta 3."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import FeatureUnion, Pipeline
    pre = None if digitos else _sin_digitos
    partes = [("palabras", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True,
                                           preprocessor=pre, max_features=200_000))]
    if chars:
        partes.append(("caracteres", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=min_df,
                                                     sublinear_tf=True, preprocessor=pre, max_features=200_000)))
    return Pipeline([("tfidf", FeatureUnion(partes)), ("ridge", Ridge(alpha=alpha))])


# ── Baseline 2: embeddings congelados + Ridge ──────────────────────────────────

def trozos(texto, palabras=180, solape=30):
    """Divide en ventanas de ~180 palabras (≈ 240 tokens, bajo el límite de 256 de MiniLM)."""
    w = texto.split()
    if len(w) <= palabras:
        return [texto]
    paso = palabras - solape
    return [" ".join(w[i:i + palabras]) for i in range(0, len(w) - solape, paso)]


def embeber(textos, modelo, batch_size=64):
    """Un vector por CV = promedio de los embeddings (normalizados) de sus trozos."""
    todos, dueno = [], []
    for i, t in enumerate(textos):
        ts = trozos(t)
        todos.extend(ts)
        dueno.extend([i] * len(ts))
    E = modelo.encode(todos, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    dueno = np.asarray(dueno)
    out = np.zeros((len(textos), E.shape[1]), dtype=np.float32)
    np.add.at(out, dueno, E)
    out /= np.bincount(dueno, minlength=len(textos))[:, None]
    return out


# ── BERT como regresor ─────────────────────────────────────────────────────────

def codificar(tokenizer, textos, max_len=512, estrategia="cabeza_cola", cola=128):
    """`cabeza`: primeros max_len−2 tokens. `cabeza_cola`: primeros (max_len−2−cola) + últimos
    `cola` (conserva el inicio del historial laboral y la educación del final)."""
    # transformers 5 ya no trae build_inputs_with_special_tokens: [CLS] … [SEP] (o BOS … EOS) a mano
    ini = [t for t in (tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id,) if t is not None]
    fin = [t for t in (tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id,) if t is not None]
    util = max_len - len(ini) - len(fin)
    ids_todos, truncados = [], 0
    for t in textos:
        ids = tokenizer(t, add_special_tokens=False, truncation=False)["input_ids"]
        if len(ids) > util:
            truncados += 1
            ids = ids[:util] if estrategia == "cabeza" else ids[:util - cola] + ids[-cola:]
        ids_todos.append(ini + ids + fin)
    return ids_todos, truncados


def texto_visto(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=True)


class DatasetRegresion:
    def __init__(self, ids, y=None):
        self.ids, self.y = ids, y

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        d = {"input_ids": self.ids[i], "attention_mask": [1] * len(self.ids[i])}
        if self.y is not None:
            d["labels"] = float(self.y[i])
        return d


def entrenar_bert(textos_tr, y_tr, textos_va, y_va, tope, salida, modelo_base="distilbert-base-uncased",
                  max_len=512, estrategia="cabeza_cola", perdida="mse", lr=3e-5, batch=16, epocas=8,
                  paciencia=2, seed=config.SEED, log=print, guardar=True):
    """Fine-tuning con AutoModelForSequenceClassification(num_labels=1, problem_type='regression').
    El target se escala a [0, 1] dividiendo por el tope; early stopping por MAE de validación (en años).

    Los checkpoints intermedios (~800 MB c/u con el optimizador) van a un directorio temporal local, no
    a Drive; al final solo se guarda el mejor modelo en `salida` (si `guardar`)."""
    import shutil
    import tempfile

    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(modelo_base)
    ids_tr, trunc_tr = codificar(tok, textos_tr, max_len, estrategia)
    ids_va, _ = codificar(tok, textos_va, max_len, estrategia)
    log(f"CVs truncados en train: {trunc_tr} de {len(ids_tr)} ({trunc_tr / len(ids_tr):.0%})")
    ds_tr = DatasetRegresion(ids_tr, np.asarray(y_tr) / tope)
    ds_va = DatasetRegresion(ids_va, np.asarray(y_va) / tope)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        modelo_base, num_labels=1, problem_type="regression")

    def metricas(ev):
        pred = np.clip(ev.predictions.reshape(-1) * tope, 0, tope)
        real = ev.label_ids.reshape(-1) * tope
        return {"mae": float(np.abs(pred - real).mean())}

    class TrainerPerdida(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            out = model(**inputs)
            pred = out.logits.squeeze(-1).float()          # pérdida en fp32 aunque el modelo corra en fp16
            labels = labels.float()
            if perdida == "huber":     # delta = 1 año en escala [0, 1]
                loss = torch.nn.functional.huber_loss(pred, labels, delta=1 / tope)
            else:
                loss = torch.nn.functional.mse_loss(pred, labels)
            return (loss, out) if return_outputs else loss

    temporal = tempfile.mkdtemp(prefix="bert_ckpt_")
    args = TrainingArguments(
        output_dir=temporal, learning_rate=lr, per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch * 2, num_train_epochs=epocas, weight_decay=0.01,
        warmup_steps=int(0.06 * epocas * -(-len(ds_tr) // batch)), eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="mae", greater_is_better=False,
        fp16=torch.cuda.is_available(), logging_steps=50, report_to="none", seed=seed,
    )
    trainer = TrainerPerdida(model=modelo, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                             data_collator=DataCollatorWithPadding(tok), compute_metrics=metricas,
                             callbacks=[EarlyStoppingCallback(early_stopping_patience=paciencia)])
    t0 = time.perf_counter()
    trainer.train()
    segundos = time.perf_counter() - t0
    if guardar:
        Path(salida).mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(salida))                 # load_best_model_at_end: es el mejor por MAE val
        tok.save_pretrained(str(salida))
    shutil.rmtree(temporal, ignore_errors=True)
    historia = [h for h in trainer.state.log_history if "eval_mae" in h]
    return trainer.model, tok, segundos, historia


def predecir_bert(modelo, tok, textos, tope, max_len=512, estrategia="cabeza_cola", batch=32, device=None):
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    modelo = modelo.to(device).eval()
    ids, _ = codificar(tok, textos, max_len, estrategia)
    preds = []
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            lote = ids[i:i + batch]
            largo = max(map(len, lote))
            pad = tok.pad_token_id or 0
            x = torch.tensor([s + [pad] * (largo - len(s)) for s in lote], device=device)
            m = torch.tensor([[1] * len(s) + [0] * (largo - len(s)) for s in lote], device=device)
            preds.append(modelo(input_ids=x, attention_mask=m).logits.squeeze(-1).float().cpu().numpy())
    return np.clip(np.concatenate(preds) * tope, 0, tope)
