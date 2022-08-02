from transformers import T5Tokenizer, T5Model
import torch
import re

class HuggingT5():
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50')
        self.model = T5Model.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        # freeze all the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def __call__(self, seq_batch, *args, **kwargs):

        if isinstance(seq_batch, str):
            seq_batch = [seq_batch]

        seq_batch = [seq.replace(" ", '') for seq in seq_batch]
        seq_batch = [' '.join(list(seq)) for seq in seq_batch]

        seq_batch = [re.sub(r"[UZOB]", "X", sequence) for sequence in seq_batch]
        ids = self.tokenizer.batch_encode_plus(seq_batch, add_special_tokens=True, padding=True)

        input_ids = torch.tensor(ids['input_ids'])
        attention_mask = torch.tensor(ids['attention_mask'])

        with torch.no_grad():
            embedding = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=input_ids)

        # For feature extraction we recommend to use the encoder embedding
        encoder_embedding = embedding[2].cpu().numpy()

        return encoder_embedding
