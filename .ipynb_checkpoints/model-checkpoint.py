from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TargetedSentimentEncoder(nn.Module):
    def __init__(self, base_model: str, device):
        super(TargetedSentimentEncoder, self).__init__()
        
        # Separate BERT encoders for query and sentence
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.query_encoder = AutoModel.from_pretrained(base_model)
        self.text_encoder = AutoModel.from_pretrained(base_model)
        
        # Freeze all layers except the last encoder stack for both networks
        #self._freeze_encoder(self.query_encoder)
        #self._freeze_encoder(self.text_encoder)
        
        self.dropout = nn.Dropout(0.3)

    def _freeze_encoder(self, encoder):
        """
        Freeze all layers except the last encoder stack.
        """
        for name, param in encoder.named_parameters():
            # Check if the layer belongs to the last encoder stack
            if 'layer' in name and 'layer.5' not in name:  # DistilBERT has 6 layers, we freeze up to layer 4
                param.requires_grad = False
    
    def forward(self, query_text, sentence_text):
        # Tokenize input
        query_inputs = self.tokenizer(query_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        sentence_inputs = self.tokenizer(sentence_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        # Get last hidden states for both query and sentence inputs
        query_hidden_states = self.query_encoder(**query_inputs).last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]
        sentence_hidden_states = self.text_encoder(**sentence_inputs).last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]

        # Apply max pooling across the sequence (dim=1)
        query_embeds, _ = torch.max(query_hidden_states, dim=1)  # Shape: [batch_size, hidden_dim]
        sentence_embeds, _ = torch.max(sentence_hidden_states, dim=1)  # Shape: [batch_size, hidden_dim]
        
        # Dropout layer
        query_embeds = self.dropout(query_embeds)
        sentence_embeds = self.dropout(sentence_embeds)
        
        return query_embeds, sentence_embeds