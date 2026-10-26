import traceback
import os

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch
import pandas as pd
from matplotlib import pyplot as plt


class TextDataset(Dataset):

    def __init__(
            self, pd_dataset: pd.DataFrame,
            char2id: dict, tag2id: dict,
            word2id: dict, inference: bool=False
    ):
        self.inference = inference
        if not self.inference:
            if not {"tokens", "lemmas", "upos"}.issubset(pd_dataset.columns):
                raise ValueError("wrong columns in dataset")
            unknown = set(pd_dataset["upos"].apply(list)\
                        .explode().unique()) - set(tag2id.keys())
            if unknown:
                raise ValueError(f"unknown UPOS: {unknown}")
            self.samples = pd_dataset[["tokens", "lemmas", "upos"]].to_dict("records")
        else:
            self.samples = pd_dataset[["tokens"]].to_dict("records")
        self.char2id = char2id
        self.tag2id = tag2id
        self.word2id = word2id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) \
    -> tuple[list[list], list[list], list[list], list[list]]:
        snt = self.samples[idx]
        unk_w = self.word2id["<UNK>"]
        words_ids = [self.word2id.get(w, unk_w) for w in snt["tokens"]]
        unk_ch = self.char2id["<UNK>"]
        chars_ids = [
            [self.char2id.get(ch, unk_ch) for ch in word]
            for word in snt["tokens"]
        ]
        if not self.inference:
            tag_ids = [self.tag2id[tag] for tag in snt["upos"]]
            lemma_char_ids = [
                [self.char2id.get(ch, unk_ch) for ch in lem] + \
                [self.char2id["<EOS>"]]
                for lem in snt["lemmas"]
            ]
        else:
            tag_ids = []
            lemma_char_ids = [[]]

        return (
            words_ids, chars_ids,
            tag_ids, lemma_char_ids
        )

class EarlyStopper:

    def __init__(self, max_iter: int, val_delta: float=0.001) -> None:
        self.max_iter = max_iter
        self.cnt = 0
        self.min_loss = float("inf")
        self.delta = val_delta

    def __call__(self, val_loss) -> bool:
        if val_loss < self.min_loss:
            self.cnt = 0
            self.min_loss = val_loss
        elif val_loss - self.delta > self.min_loss:
            self.cnt += 1
            if self.cnt > self.max_iter:
                return True
        return False
    

def collate_fn(
    batch, pad_word_id: int = 0,
    pad_char_id: int = 0, pad_tag_id: int = -100,
    pad_lemma_id: int = -100,
):
    word_ids_list, char_ids_list, tag_ids_list, lemma_char_ids_list = zip(*batch)

    word_ids = pad_sequence(
        [torch.tensor(w, dtype=torch.long) for w in word_ids_list],
        batch_first=True,
        padding_value=pad_word_id,
    )
    tag_ids = pad_sequence(
        [torch.tensor(t, dtype=torch.long) for t in tag_ids_list],
        batch_first=True,
        padding_value=pad_tag_id
    )
    bsize = len(char_ids_list)
    cnt_words = max(len(sent) for sent in char_ids_list)
    word_len = max(
        (len(word) for sent in char_ids_list for word in sent)
    )
    char_ids = torch.full(
        (bsize, cnt_words, word_len),
        pad_char_id, dtype=torch.long
    )
    for i, snt in enumerate(char_ids_list):
        for j, word in enumerate(snt):
            if word:
                char_ids[i, j, :len(word)] = torch.tensor(word, dtype=torch.long)

    lemma_len = max(
        (len(lem) for sent in lemma_char_ids_list for lem in sent)
    )
    lemma_char_ids = torch.full(
        (bsize, cnt_words, lemma_len),
        pad_lemma_id, dtype=torch.long
    )
    for i, sent in enumerate(lemma_char_ids_list):
        for j, lem in enumerate(sent):
            if lem:
                lemma_char_ids[i, j, :len(lem)] = torch.tensor(lem, dtype=torch.long)

    return word_ids, char_ids, tag_ids, lemma_char_ids


def compute_loss(
        tag_loss, lemma_loss, log_sigma_tag,
        log_sigma_lemma
):
    mn = -1.7
    mx = 1.7
    # tag_w = torch.exp(-log_sigma_tag.clamp(mn, mx))
    # lemma_w = torch.exp(-log_sigma_lemma.clamp(mn, mx))
    # return (
    #     tag_w * tag_loss + log_sigma_tag.clamp(mn, mx)
    #     + lemma_w * lemma_loss + log_sigma_lemma.clamp(mn, mx)
    # )
    tag_w = 1.0
    lemma_w = 0.5
    return (tag_w * tag_loss + lemma_w * lemma_loss)


def nn_val(
        model, loader, tag_crit, lemma_crit,
        device="cuda"
):
    """
    Returns:
        loss_value (float)
    """
    model.to(device)
    model.eval()
    loss_value = 0.0
    tag_loss_val = 0.0
    lem_loss_val = 0.0
    with torch.no_grad():
        for batch in loader:
            word_ids, char_ids, tag_ids, lemma_char_ids = [
                x.to(device) for x in batch
            ]
            tag_logits, _, lemma_logits = model(
                word_ids, char_ids,
                lemma_char_ids=lemma_char_ids, tag_ids=tag_ids,
            )
            tag_loss = tag_crit(
                tag_logits.reshape(-1, tag_logits.size(-1)),
                tag_ids.reshape(-1),
            )
            tag_loss_val += tag_loss.item()
            lemma_lengths = (lemma_char_ids != -100).sum()
            lemma_loss = lemma_crit(
                lemma_logits.reshape(-1, lemma_logits.size(-1)),
                lemma_char_ids.reshape(-1),
            ) / lemma_lengths.clamp(min=1)
            lem_loss_val += lemma_loss.item()
            loss = compute_loss(
                tag_loss, lemma_loss,
                model.log_sigma_tag,
                model.log_sigma_lemma
            )
            loss_value += loss.item()
        loss_value /= len(loader)
        tag_loss_val /= len(loader)
        lem_loss_val /= len(loader)

    return loss_value, tag_loss_val, lem_loss_val


def nn_predict(
        model, loader, id2char: dict,
        device="cuda", max_lemma_len: int = 20,
        pad_id: int = 0, unk_id: int = 1
):
    """
    Returns:
        all_tag_preds: list[list[int]] UPOS-id for each word
        all_lemma_preds: list[list[str]] strings are lemmas
    """
    model.to(device)
    model.eval()
    all_tag_preds = []
    all_lemma_preds = []
    unk_char = id2char.get(unk_id, "<UNK>")
    with torch.no_grad():
        for batch in loader:
            word_ids, char_ids, *other = [
                x.to(device) for x in batch
            ]
            word_mask = (word_ids != pad_id)
            tag_logits, lemma_ids, _ = model(
                word_ids, char_ids, max_lemma_len=max_lemma_len
            )
            tag_argmax = tag_logits.argmax(dim=-1)
            for i in range(word_ids.size(0)):
                L = int(word_mask[i].sum().item())
                all_tag_preds.append(tag_argmax[i, :L].tolist())
                lemma_row = []
                for j in range(L):
                    ids = lemma_ids[i, j].tolist()
                    stop_positions = [
                        ids.index(s) for s in (model.eos_id, pad_id) if s in ids
                    ]
                    if stop_positions:
                        ids = ids[:min(stop_positions)]
                    chars = [
                        id2char.get(c, unk_char)
                        for c in ids
                    ]
                    lemma_row.append("".join(chars))
                all_lemma_preds.append(lemma_row)

    return all_tag_preds, all_lemma_preds


def nn_train(
        model, train_loader, tag_crit, lemma_crit,
        optimizer, val_loader=None, id2char=None,
        num_epoch=100, device="cuda",
        print_progress=True, early_stopper=lambda x: False
) -> tuple:
    """
    Returns:
        out (tuple):
            1) model
            2) optimizer
            3) train_loss_lst
    """
    best_loss = 1e10
    num_batches = len(train_loader)
    model.to(device)
    best_params = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
    }
    if val_loader is not None and (
            tag_crit is None or 
            lemma_crit is None or
            id2char is None):
        raise ValueError(
            "if val_loader is not None it's necessary "
            "to provide tag_crit, lemma_crit, id2char"
        )
    train_loss_lst = []
    val_loss_lst = [1e10]
    train_tag_loss = []
    train_lem_loss = []
    val_tag_loss = []
    val_lem_loss = []
    try:
        for ep in range(num_epoch):
            ep_tag_loss = 0.0
            ep_lem_loss = 0.0
            model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                word_ids, char_ids, tag_ids, lemma_char_ids = [
                    x.to(device) for x in batch
                ]
                with torch.no_grad():
                    lemma_lengths = (lemma_char_ids != -100).sum()
                tag_logits, _, lemma_logits = model(
                    word_ids, char_ids, lemma_char_ids, tag_ids
                )
                tag_loss = tag_crit(
                    tag_logits.reshape(-1, tag_logits.size(-1)),
                    tag_ids.reshape(-1)
                )
                ep_tag_loss += tag_loss.item()
                lemma_loss = lemma_crit(
                    lemma_logits.reshape(-1, lemma_logits.size(-1)),
                    lemma_char_ids.reshape(-1)
                ) / lemma_lengths.clamp(min=1)
                ep_lem_loss += lemma_loss.item()
                loss = compute_loss(
                    tag_loss, lemma_loss,
                    model.log_sigma_tag,
                    model.log_sigma_lemma
                )
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss_lst.append(train_loss / num_batches)
            ep_tag_loss /= num_batches
            ep_lem_loss /= num_batches
            train_tag_loss.append(ep_tag_loss)
            train_lem_loss.append(ep_lem_loss)
            show_tag_loss = ep_tag_loss
            show_lem_loss = ep_lem_loss
            if val_loader is not None:
                val_loss, vtag_loss, vlem_loss = nn_val(
                    model, val_loader, tag_crit,
                    lemma_crit, device
                )
                val_loss_lst.append(val_loss)
                val_tag_loss.append(vtag_loss)
                val_lem_loss.append(vlem_loss)
                show_tag_loss = vtag_loss
                show_lem_loss = vlem_loss
            if print_progress:
                print(
                    f"end epoch {ep:5d}|train loss {train_loss_lst[-1]:.5f}"
                    f"|val loss {val_loss_lst[-1]:.5f}"
                    f"|σ_tag {model.log_sigma_tag.item():+.3f}"
                    f"|σ_lemma {model.log_sigma_lemma.item():+.3f}"
                    f"|tag_loss {show_tag_loss:.4f}"
                    f"|lem_loss {show_lem_loss:.4f}"
                )
            if val_loader is not None:
                raw_loss = val_tag_loss[-1] + val_lem_loss[-1]
            else:
                raw_loss = train_tag_loss[-1] + train_lem_loss[-1]
            if raw_loss < best_loss:
                best_loss = raw_loss
                best_params = {
                    k: v.detach().clone()
                    for k, v in model.state_dict().items()
                }
            if early_stopper(raw_loss):
                print("early stop by raw loss")
                break
    except BaseException:
        print(traceback.format_exc())
    
    model.load_state_dict(best_params)
    return (
        model, optimizer,
        {
            "gen_loss": train_loss_lst,
            "tag_loss": train_tag_loss,
            "lem_loss": train_lem_loss
        },
        {
            "gen_loss": val_loss_lst[1:],
            "tag_loss": val_tag_loss,
            "lem_loss": val_lem_loss
        }
    )


def token_accuracy(gold_lexes, pred_lexes):
    correct = 0
    total = 0
    for pred_sent, gold_sent in zip(pred_lexes, gold_lexes):
        for pred_lem, gold_lem in zip(pred_sent, gold_sent):
            correct += int(pred_lem == gold_lem)
            total += 1
    return correct / total


def plot_results(
        train_loss: dict, val_loss: dict|None=None,
        metric=None, save_fig=False, model_name="model"
) -> None:
    os.makedirs("images", exist_ok=True)
    x = list(range(1, len(train_loss["gen_loss"]) + 1))
    if metric is not None:
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(11, 9))
        axs[0].plot(x, metric, marker=".")
        axs[0].grid(axis="x")
        axs[0].set_title("Metric")
        if val_loss is not None and len(val_loss["gen_loss"]) > 0:
            axs[1].plot(x, val_loss["gen_loss"], marker=".", label="val loss")
            axs[2].plot(x, val_loss["tag_loss"], marker=".", label="val tag loss")
            axs[3].plot(x, val_loss["lem_loss"], marker=".", label="val lemm loss")
        axs[1].plot(x, train_loss["gen_loss"], marker=".", label="train loss")
        axs[1].set_title("Loss")
        axs[1].grid(axis="x")
        axs[1].legend()
        axs[2].plot(x, train_loss["tag_loss"], marker=".", label="train tag loss")
        axs[2].legend()
        axs[2].grid(axis="x")
        axs[2].set_title("Tag loss")
        axs[3].plot(x, train_loss["lem_loss"], marker=".", label="train lemm loss")
        axs[3].legend()
        axs[3].grid(axis="x")
        axs[3].set_title("Lemma loss")
        axs[3].set_xlabel("num epoch")
        fig.tight_layout()
    else:
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(11, 7))
        if val_loss is not None and len(val_loss["gen_loss"]) > 0:
            axs[0].plot(x, val_loss["gen_loss"], marker=".", label="val loss")
            axs[1].plot(x, val_loss["tag_loss"], marker=".", label="val tag loss")
            axs[2].plot(x, val_loss["lem_loss"], marker=".", label="val lemm loss")
        axs[0].plot(x, train_loss["gen_loss"], marker=".", label="train loss")
        axs[0].set_title("Loss")
        axs[0].grid(axis="x")
        axs[0].legend()
        axs[1].plot(x, train_loss["tag_loss"], marker=".", label="train tag loss")
        axs[1].legend()
        axs[1].grid(axis="x")
        axs[1].set_title("Tag loss")
        axs[2].plot(x, train_loss["lem_loss"], marker=".", label="train lemm loss")
        axs[2].legend()
        axs[2].grid(axis="x")
        axs[2].set_title("Lemma loss")
        axs[2].set_xlabel("num epoch")
        fig.tight_layout()
    if save_fig:
        plt.savefig(f"images/{model_name}.png")
    else:
        plt.show()


def prep_raw_txt(fname) -> pd.DataFrame:
    with open(fname, "r", encoding="utf-8") as f:
        words = []
        symbs = "йцукенгшщзхъфывапролджэячсмитьбю"
        symbs += symbs.upper()
        symbs += ".,?!1234567890"
        symbs = set(symbs)
        rows = f.readlines()
        for r in rows:
            r = r.replace(".", " ").replace("?", " ")
            r = r.replace("!", " ").replace(",", " ")
            r = r.replace("ё", "е").replace("Ё", "Е")
            r = r.lower()
            words.append([
                w for w in r.split()
                if symbs.issuperset(w)
            ])
    return pd.DataFrame({"tokens": words})


def transform_out(
        tokens: list[list[str]],
        tags: list[list[int]],
        lemmas: list[list[str]],
        id2tag: dict, out_fname: str
):
    with open(out_fname, "w", encoding="utf-8") as file:
        for sent, sent_tags, sent_lemmas in zip(tokens, tags, lemmas):
            combined = []
            for word, tag, lem in zip(sent, sent_tags, sent_lemmas):
                combined.append(word + "{" + lem + "=" + id2tag[tag] + "}")
            comb = " ".join(combined) + "\n"
            file.write(comb)
