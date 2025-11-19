import torch
import requests
import re
import requests
from bs4 import BeautifulSoup
RESET = "\033[0m"
BLUE_ = "\033[94m"

def load_weights(pth_path):

    checkpoint = torch.load(pth_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            params = checkpoint["model"]
        elif "state_dict" in checkpoint:
            params = checkpoint["state_dict"]
        else:
            params = checkpoint  
    else:
        raise ValueError("Unexpected checkpoint format in .pth file")

    print(f"Loaded {len(params)} tensors from {pth_path}")
    return params


def context_hunter(prompt, n_results=1):
    url = "https://serpapi.com/search.json"
    params = {
        "q": prompt,
        "api_key": "",  
        "num": n_results
    }

    response = requests.get(url, params=params)
    data = response.json()

    contexts = []
    links = []

    for item in data.get("organic_results", [])[:n_results]:
        snippet = item.get("snippet")
        link = item.get("link")
        if snippet:
            contexts.append(snippet)
        if link:
            links.append(link)

    if "answer_box" in data:
        contexts.append(str(data["answer_box"]))
    if "knowledge_graph" in data:
        contexts.append(str(data["knowledge_graph"]))

    for link in links:
        try:
            page = requests.get(link, timeout=10)
            soup = BeautifulSoup(page.text, "html.parser")

            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            page_text = " ".join(paragraphs)

            if page_text:
                contexts.append(page_text[:1000])  
        except Exception as e:
            print(f"Skipping {link} due to error: {e}")

    return "\n\n".join(contexts)


def cleaner(raw_data):

    if isinstance(raw_data, (dict, list)):
        raw_data = str(raw_data)

    elif not isinstance(raw_data, str):
        raw_data = str(raw_data)

    cleaned = re.sub(r"\{.*?\}", "", raw_data)
    cleaned = re.sub(r"\[.*?\]", "", cleaned)

    cleaned = re.sub(r"https?://\S+", "", cleaned)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"&[a-z]+;", "", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def weight_injector(my_model, weights_pth):

    params = load_weights(weights_pth)
    device = next(my_model.parameters()).device
    dtype = torch.bfloat16

    my_model.tok_embedding.weight.data.copy_(
        params["tok_embeddings.weight"].to(device=device, dtype=dtype)
    )

    num_layers = len(my_model.layers)
    for i in range(num_layers):
        layer = my_model.layers[i]

        layer.Attention.W_Q.weight.data.copy_(
            params[f"layers.{i}.attention.wq.weight"].to(device=device, dtype=dtype)
        )
        layer.Attention.W_K.weight.data.copy_(
            params[f"layers.{i}.attention.wk.weight"].to(device=device, dtype=dtype)
        )
        layer.Attention.W_V.weight.data.copy_(
            params[f"layers.{i}.attention.wv.weight"].to(device=device, dtype=dtype)
        )
        layer.Attention.wo.weight.data.copy_(
            params[f"layers.{i}.attention.wo.weight"].to(device=device, dtype=dtype)
        )

        if hasattr(layer.Attention.wo, "bias") and f"layers.{i}.attention.wo.bias" in params:
            layer.Attention.wo.bias.data.copy_(
                params[f"layers.{i}.attention.wo.bias"].to(device=device, dtype=dtype)
            )

        layer.FeedForward.w1.weight.data.copy_(
            params[f"layers.{i}.feed_forward.w1.weight"].to(device=device, dtype=dtype)
        )
        layer.FeedForward.w3.weight.data.copy_(
            params[f"layers.{i}.feed_forward.w3.weight"].to(device=device, dtype=dtype)
        )
        layer.FeedForward.w2.weight.data.copy_(
            params[f"layers.{i}.feed_forward.w2.weight"].to(device=device, dtype=dtype)
        )

        layer.Attention_Norm.weight.data.copy_(
            params[f"layers.{i}.attention_norm.weight"].to(device=device, dtype=dtype)
        )
        layer.FFN_Norm.weight.data.copy_(
            params[f"layers.{i}.ffn_norm.weight"].to(device=device, dtype=dtype)
        )

    my_model.norm.weight.data.copy_(
        params["norm.weight"].to(device=device, dtype=dtype)
    )

    out_w = params["output.weight"]
    if my_model.output.weight.shape == out_w.shape:
        my_model.output.weight.data.copy_(out_w.to(device=device, dtype=dtype))
    elif my_model.output.weight.shape[::-1] == out_w.shape:
        my_model.output.weight.data.copy_(out_w.T.to(device=device, dtype=dtype))
    else:
        raise ValueError(f"Output weight shape mismatch: model={my_model.output.weight.shape}, ckpt={out_w.shape}")

    print(f"Model successfully loaded to: {device}")


def format_prompt(role, content):
    return (
        f"<|start_header_id|>{role}<|end_header_id|>\n"
        f"{content.strip()}\n"
        "<|eot_id|>\n"
    )


def Chat(
    model, tokenizer, conversation_tokens, max_new_tok,
    top_k=50, context_len=2048, temp=0.9,
    eos_id=128001, eot_id=128009, eom_id=128008):

    model.eval()
    idx = conversation_tokens
    assert idx.ndim == 2, "conversation_tokens must be shape (batch, seq_len)"
    if idx.shape[0] != 1:
        raise NotImplementedError("Chat currently supports batch size 1 only (provide a single conversation).")

    initial_len = idx.shape[1]

    if temp <= 0:
        raise ValueError("temp must be greater than 0")

    if initial_len > context_len:
        idx_cond = idx[:, -context_len:]
        start_pos = 0
    else:
        idx_cond = idx
        start_pos = 0

    generated_text = ""
    current_pos = start_pos + idx_cond.shape[1]

    for tok_no in range(max_new_tok):
        if tok_no == 0:
            with torch.inference_mode():
                logits = model(idx_cond, start_pos)
        else:
            idx_cond_step = idx[:, -1:].to(idx.device)
            with torch.inference_mode():
                logits = model(idx_cond_step, current_pos)
            current_pos += 1

        logits = logits[:, -1, :] 

        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_vals, _ = torch.topk(logits, top_k, dim=-1)
            min_val = top_k_vals[:, -1:].expand_as(logits)
            logits = torch.where(logits < min_val, float("-inf"), logits)

        logits = logits / float(temp)

        probs = torch.softmax(logits, dim=-1)
        preds = torch.multinomial(probs, 1) 
        next_tok = int(preds[0, 0])

        if next_tok in {eos_id, eot_id, eom_id}:
            print(f"\n\n[Stop token {next_tok} reached at step {tok_no+1}]\n")
            idx = torch.cat([idx, preds], dim=1)
            break

        idx = torch.cat([idx, preds], dim=1)

        token_text = tokenizer.decode([next_tok])
        generated_text += token_text

        if tok_no == 0:
            try:
                print(f"{BLUE_}Blue:{RESET} ", end="", flush=True)
            except Exception:
                print("Blue: ", end="", flush=True)

        print(token_text, end="", flush=True)

    if tok_no == max_new_tok - 1:
        print(f"\n\n[MAX NEW TOKENS REACHED!]\n")

    assistant_tokens = idx[0, initial_len:].tolist()
    assistant_text = tokenizer.decode(assistant_tokens)
    assistant_text = assistant_text.split("<|eot_id|>")[0].strip()

    return idx, assistant_text
