from torch.nn import (
    Module,
    LSTM,
    Linear,
    Embedding,
    Dropout
)
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch

class Model(Module):

    def __init__(
            self, vocab_size, char_vocab_size,
            tagset_size, word_emb_dim,
            char_emb_dim, hidden_dim: int=128,
            gen_enc_n_layer: int=3, dropout: float=0.3,
            sos_id: int=2, eos_id: int=3, pad_id: int=0
    ):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        ## weight params for loss
        self.log_sigma_tag = torch.nn.Parameter(torch.zeros(1))
        self.log_sigma_lemma = torch.nn.Parameter(torch.zeros(1))
        char_rnn_hidden = char_emb_dim // 2
        char_out_dim = 2 * char_rnn_hidden
        ## embedding
        self.word_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=word_emb_dim,
            padding_idx=0
        )
        self.char_embedding = Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=char_emb_dim,
            padding_idx=0
        )
        self.tag_embedding = Embedding(tagset_size, hidden_dim)
        ## char encoder
        self.char_rnn = LSTM(
            input_size=char_emb_dim, hidden_size=char_rnn_hidden,
            bidirectional=True, batch_first=True
        )
        ## general encoder
        self.general_rnn = LSTM(
            input_size=word_emb_dim + char_out_dim,
            hidden_size=hidden_dim // 2,
            num_layers=gen_enc_n_layer,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = Dropout(dropout)
                                # hidden state + char representation
        self.attn_score = Linear(hidden_dim + char_out_dim, 1)
        ## tag decoder
        self.tag_classifier = Linear(hidden_dim, tagset_size)
        ## lemm decoder
                                #  word context + tag embedding
        self.decoder_init = Linear(hidden_dim + hidden_dim, hidden_dim)
        self.decoder_rnn = LSTM(
            # embed prev generated char + attn context
            char_emb_dim + char_out_dim,
            hidden_dim,
            batch_first=True
        )
        self.decoder_out = Linear(hidden_dim, char_vocab_size)

    def encode(self, word_ids, char_ids):
        batch_size, sent_len = word_ids.size()
        word_len = char_ids.size(2)
        word_embed = self.word_embedding(word_ids) # [batch_size, sent_len, word_emb_dim]
        ## each word as independend example
        char_flat = char_ids.view(batch_size * sent_len, word_len)
        char_embed = self.char_embedding(char_flat)
        char_out, _ = self.char_rnn(char_embed)

        char_mask = (char_flat != 0) # skip pad chars
        ## masked mean-pooling across chars
        char_repr = (char_out * char_mask.unsqueeze(-1)).sum(1) / \
            char_mask.sum(1, keepdim=True).clamp(min=1.0)
        char_repr = char_repr.view(batch_size, sent_len, -1)

        combined = torch.cat([word_embed, char_repr], dim=-1)
        ## len of each sentence without pad in batch
        lengths = (word_ids != 0).sum(dim=1).clamp(min=1).cpu()
        ## pack batch for rnn
        packed = pack_padded_sequence(
            combined, lengths,
            batch_first=True,
            enforce_sorted=False
        )
        packed_out, _ = self.general_rnn(packed)
        ## unpack batch after rnn
        context, _ = pad_packed_sequence(
            packed_out, batch_first=True,
            total_length=sent_len
        )

        return self.dropout(context), char_out, char_mask

    def attend(self, query, keys, key_mask):
        ## query - hidden decoder state
        ## keys - char representation for word
        ## key_mask - boolean mask
        q = query.unsqueeze(1).expand(-1, keys.size(1), -1)
        score = self.attn_score(torch.cat([q, keys], dim=-1)).squeeze(-1)
        ## padded chars ignored by softmax
        score = score.masked_fill(~key_mask, -1e9)
        attn = torch.softmax(score, dim=-1)
        ## attn context
        ctx  = (attn.unsqueeze(-1) * keys).sum(1)
        return ctx

    def _decode_step(
            self, prev_char, hidden, char_seq, char_mask
    ):
        h_last = hidden[0][-1]
        ctx = self.attend(h_last, char_seq, char_mask)

        dec_in = torch.cat(
            [self.char_embedding(prev_char), ctx], dim=-1
        ).unsqueeze(1)
        out, hidden = self.decoder_rnn(dec_in, hidden)
        logits = self.decoder_out(out).squeeze(1)
        return logits, hidden

    def forward(
            self, word_ids, char_ids, lemma_char_ids=None,
            tag_ids=None, max_lemma_len: int = 20
    ):
        batch_size, sent_len = word_ids.size()
        context, char_seq, char_mask = self.encode(word_ids, char_ids)
        tag_logits = self.tag_classifier(context)

        if tag_ids is not None: ## teacher-forcing
            decode_tags = tag_ids.clamp(0, self.tag_embedding.num_embeddings - 1)
        else:
            decode_tags = tag_logits.argmax(dim=-1)
        tag_embed = self.tag_embedding(decode_tags)
        ## add tag to lemm decoder
        init = torch.cat([context, tag_embed], dim=-1)
        h = torch.tanh(self.decoder_init(init))
        h = h.reshape(batch_size * sent_len, self.hidden_dim).unsqueeze(0).contiguous()
        c = torch.zeros_like(h)
        hidden = (h, c)
        N = batch_size * sent_len
        if lemma_char_ids is None: ## inference
            ids = self._greedy_decode(
                hidden, char_seq, char_mask,
                N, batch_size, sent_len, max_lemma_len
            )
            return tag_logits, ids, None
        ## training
        lemma_len = lemma_char_ids.size(2)
        lemma_flat = lemma_char_ids.view(N, lemma_len)
        prev_char = torch.full(
            (N,), self.sos_id, dtype=torch.long, device=word_ids.device
        )
        logits_seq = []
        ## generate next lemma's char based on prev char
        for t in range(lemma_len):
            step_logits, hidden = self._decode_step(
                prev_char, hidden, char_seq, char_mask
            )
            logits_seq.append(step_logits.unsqueeze(1))
            prev_char = lemma_flat[:, t].clamp(min=0)
        lemma_logits = torch.cat(logits_seq, dim=1)
        lemma_logits = lemma_logits.view(batch_size, sent_len, lemma_len, -1)
        return tag_logits, None, lemma_logits

    def _greedy_decode(
            self, hidden, char_seq, char_mask,
            N, batch_size, sent_len, max_len
    ):
        with torch.no_grad():
            prev_char = torch.full(
                (N,), self.sos_id, dtype=torch.long,
                device=char_seq.device
            )
            finished = torch.zeros(
                N, dtype=torch.bool, device=char_seq.device
            )
            outputs = []
            for _ in range(max_len):
                logits, hidden = self._decode_step(
                    prev_char, hidden, char_seq, char_mask
                )
                nxt = logits.argmax(dim=-1)
                nxt = torch.where(
                    finished,
                    torch.full_like(nxt, self.pad_id),
                    nxt,
                )
                outputs.append(nxt.unsqueeze(1))
                finished = finished | (nxt == self.eos_id)
                prev_char = nxt
                if finished.all():
                    break
            ids = torch.cat(outputs, dim=1)
            if ids.size(1) < max_len:
                pad = torch.full(
                    (N, max_len - ids.size(1)), self.pad_id,
                    dtype=torch.long, device=ids.device
                )
                ids = torch.cat([ids, pad], dim=1)
            
            return ids.view(batch_size, sent_len, max_len)
