import torch
from torch import nn
from torch.utils import checkpoint
import transformers
import types

MODEL_SIZE = "gpt2-medium"
CHECKPOINTING_START = 1

TEMPERATURE = 0.6
NUCLEUS_THRESHOLD = 0.3


class PsuedoSelf(nn.Module):
    def __init__(self, attention, context_size, key=None, value=None):
        super().__init__()

        self.attention = attention._attn
        self.n_head = attention.n_head
        self.split = attention.split_size // attention.n_head

        self.key_transform = key or nn.Linear(context_size, attention.split_size)
        self.value_transform = value or nn.Linear(context_size, attention.split_size)

        attention._attn = types.MethodType(self, attention)

    def set_context(self, context):
        self.context_key = self.key_transform.forward(context)
        self.context_key = self.context_key.view(-1, self.n_head, self.split).unsqueeze(3)

        self.context_value = self.value_transform.forward(context)
        self.context_value = self.context_value.view(-1, self.n_head, self.split).unsqueeze(2)

    def forward(self, _, query, key, value, attention_mask=None, head_mask=None):
        attention_mask = torch.cat((torch.zeros(attention_mask.shape[0], 1, 1, 1, device=attention_mask.device), attention_mask), dim=-1)
        return self.attention(query, torch.cat((self.context_key, key), dim=3), torch.cat((self.context_value, value), dim=2), attention_mask, head_mask)


class CheckpointBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        return (checkpoint.checkpoint(lambda *x: self.block(*x, **kwargs)[0], *args), None)


class Decoder(nn.Module):
    def __init__(self, input_size, **kwargs):
        super().__init__()

        self.gpt = transformers.GPT2LMHeadModel.from_pretrained(MODEL_SIZE)
        self.psuedo_self = PsuedoSelf(self.gpt.transformer.h[0].attn, input_size, **kwargs)

        for i in range(CHECKPOINTING_START, len(self.gpt.transformer.h)):
            self.gpt.transformer.h[i] = CheckpointBlock(self.gpt.transformer.h[i])

    def parameters(self):
        return self.psuedo_self.parameters()

    def forward(self, context, labels=None, length=None):
        self.psuedo_self.set_context(context)

        if labels is not None:
            return self.gpt(input_ids=torch.max(labels[0], torch.zeros_like(labels[0])), labels=labels[0], attention_mask=labels[1])
        elif length is not None:
            with torch.no_grad():
                past_tokens = torch.zeros(context.shape[0], 1, dtype=torch.long, device=context.device)
                attention_mask = torch.zeros(context.shape[0], 1, device=context.device)

                for _ in range(length):
                    scores = nn.functional.softmax(self.gpt(input_ids=past_tokens, attention_mask=attention_mask)[0][:, -1, :] / TEMPERATURE, dim=-1)

                    sort, positions = torch.sort(scores)
                    cutoff_tokens = torch.cumsum(sort, dim=-1) < NUCLEUS_THRESHOLD
                    scores[cutoff_tokens.scatter(dim=1, index=positions, src=cutoff_tokens)] = 0

                    token = torch.multinomial(scores, num_samples=1)
                    past_tokens = torch.cat((past_tokens, token), dim=-1)
                    attention_mask = torch.cat((attention_mask, torch.ones(context.shape[0], 1, device=context.device)), dim=-1)

            return past_tokens[:, 1:]
