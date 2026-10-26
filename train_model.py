import json

import pandas as pd
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch

from model import Model
from utils import (
    TextDataset,
    collate_fn,
    nn_train,
    nn_predict,
    token_accuracy,
    plot_results,
    EarlyStopper,
    prep_raw_txt,
    transform_out
)


if __name__ == "__main__":
    with open("data/char2id.json", "r", encoding="utf-8") as f:
        char2id = json.load(f)
    with open("data/word2id.json", "r", encoding="utf-8") as f:
        word2id = json.load(f)
    with open("data/tag2id.json", "r", encoding="utf-8") as f:
        tag2id = json.load(f)
    with open("data/id2char.json", "r", encoding="utf-8") as f:
        id2char = json.load(f)
        id2char = {int(k): v for k, v in id2char.items()}
    train_data = pd.read_pickle("data/train_prepr.pcl").iloc[::6]
    val_data = pd.read_pickle("data/val_prepr.pcl").iloc[::4]
    train_dataset = TextDataset(train_data, char2id, tag2id, word2id)
    val_dataset = TextDataset(val_data, char2id, tag2id, word2id)
    train_loader = DataLoader(train_dataset, 16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, 16, shuffle=False, collate_fn=collate_fn)
    model = Model(
        max(word2id.values()) + 1, max(char2id.values()) + 1,
        max(tag2id.values()) + 1, word_emb_dim=50, char_emb_dim=40,
        sos_id=char2id["<SOS>"], eos_id=char2id["<EOS>"], pad_id=char2id["<PAD>"]
    )

    def train_cycle(
            tag2id, id2char, model,
            train_loader, val_loader,
            val_data
    ):
        optimizer = Adam(model.parameters(), lr=3e-4)
        tag_crit = CrossEntropyLoss(ignore_index=-100, reduction="mean")
        lemma_crit = CrossEntropyLoss(ignore_index=-100, reduction="sum")
        early_stopper = EarlyStopper(10, 0.01)
        print("==================Start training====================")
        train_model, train_optim, train_loss, val_loss = nn_train(
            model, train_loader, tag_crit, lemma_crit,
            optimizer, device="cuda", num_epoch=100,
            val_loader=val_loader, id2char=id2char,
            early_stopper=early_stopper
        )
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": train_optim.state_dict()
        }, f"models/model.pt")
        plot_results(
            train_loss, val_loss, save_fig=True
        )
        all_tag_preds, all_lemma_preds = nn_predict(
            train_model, val_loader, id2char, "cpu"
        )
        print("all_tag_preds", all_tag_preds[:3])
        print("all_lemma_preds", all_lemma_preds[:3])
        print("lemma", val_data["lemmas"].iloc[:3])
        print("lemma accuracy", token_accuracy(
            val_data["lemmas"], all_lemma_preds))
        print("tag accuracy", token_accuracy(
            [
                [tag2id[t] for t in sent]
                for sent in val_data["upos"]
            ],
            all_tag_preds
        ))


    def test_accuracy(
            model, model_fname, data_fname, id2char, char2id, tag2id,
            word2id
    ):
        data = pd.read_pickle(data_fname)
        dataset = TextDataset(data, char2id, tag2id, word2id)
        loader = DataLoader(dataset, 8, shuffle=False, collate_fn=collate_fn)
        checkpoint = torch.load(model_fname, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        all_tag_preds, all_lemma_preds = nn_predict(
            model, loader, id2char, "cpu"
        )
        print("lemma accuracy", token_accuracy(
            data["lemmas"], all_lemma_preds
        ))
        print("tag accuracy", token_accuracy(
            [
                [tag2id[t] for t in sent]
                for sent in data["upos"]
            ],
            all_tag_preds
        ))


    def new_data_test(
            model, model_fname, data_fname,
            out_fname,
            char2id, word2id, tag2id, id2char
    ):
        checkpoint = torch.load(model_fname, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        id2tag = {val: key for key, val in tag2id.items()}
        data = prep_raw_txt(data_fname)
        dataset = TextDataset(data, char2id, tag2id, word2id, True)
        loader = DataLoader(
            dataset, batch_size=8, shuffle=False,
            collate_fn=collate_fn
        )
        all_tag_preds, all_lemma_preds = nn_predict(
            model, loader, id2char, "cpu"
        )
        transform_out(
            data["tokens"], all_tag_preds, all_lemma_preds,
            id2tag, out_fname
        )


    # train_cycle(
    #     tag2id, id2char, model, train_loader,
    #     val_loader, val_data
    # )

    # test_accuracy(
    #     model, "models/model_prep_token.pt",
    #     "data/val_prepr.pcl",
    #     id2char, char2id, tag2id, word2id
    # )

    # test_accuracy(
    #     model, "models/model_const.pt",
    #     "data/val_prepr.pcl",
    #     id2char, char2id, tag2id, word2id
    # )

    new_data_test(
        model, "models/model_const.pt", "test.txt",
        "out.txt", char2id, word2id, tag2id, id2char
    )

    # test_accuracy(
    #     model, "models/model_const.pt",
    #     "data/test_prepr.pcl",
    #     id2char, char2id, tag2id, word2id
    # )
    
