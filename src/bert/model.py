import torch
import torch.nn as nn
from transformers import DistilBertModel

from preprocessing import MODEL_NAME


# ============================================================
# CONFIGURATION
# ============================================================

DENSE_UNITS  = 128   # hidden size for the fully-connected layers in both branches
DROPOUT_RATE = 0.3   # dropout probability applied after every dense layer


# ============================================================
# MODEL
# ============================================================

class BertTicketClassifier(nn.Module):
    """
    Dual-input model with fine-tuned DistilBERT and two output heads:

    input_ids + attention_mask ──→ DistilBERT ──→ [CLS] ──→ Dense ──┐
                                                                      ├──→ Dense ──→ category head
    structured_input ──→ Dense ──→ Dense ───────────────────────────┘         └──→ subcategory head

    The model processes ticket text through DistilBERT to capture semantic
    meaning, while a separate MLP branch handles structured/tabular features
    (e.g. priority, severity, customer tier). Both branches are merged via
    concatenation before feeding into two independent classification heads
    for predicting ticket category and subcategory.
    """

    def __init__(self, num_structured_features: int,
                 num_categories: int, num_subcategories: int):
        super().__init__()

        # ---- text branch (frozen DistilBERT) ----
        # Load a pre-trained DistilBERT; weights are frozen to speed up training on CPU
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)
        for param in self.bert.parameters():
            param.requires_grad = False
        bert_hidden = self.bert.config.hidden_size  # 768 for distilbert-base

        # Project the 768-dim [CLS] representation down to DENSE_UNITS
        self.text_dense = nn.Sequential(
            nn.Linear(bert_hidden, DENSE_UNITS),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )

        # ---- structured branch ----
        # Two-layer MLP that learns representations from tabular features
        # (one-hot encoded, label-encoded, numeric, and binary columns)
        self.struct_branch = nn.Sequential(
            nn.Linear(num_structured_features, DENSE_UNITS),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(DENSE_UNITS, DENSE_UNITS),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )

        # ---- shared layer after merge ----
        # Fuses the text and structured representations (concatenated → 2*DENSE_UNITS)
        # into a single DENSE_UNITS-dim vector used by both output heads
        self.shared = nn.Sequential(
            nn.Linear(DENSE_UNITS * 2, DENSE_UNITS),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )

        # ---- output heads ----
        # Each head is a simple linear layer producing raw logits;
        # softmax is handled inside CrossEntropyLoss during training
        self.category_head = nn.Linear(DENSE_UNITS, num_categories)
        self.subcategory_head = nn.Linear(DENSE_UNITS, num_subcategories)

    def forward(self, input_ids, attention_mask, structured):
        # --- Text branch ---
        # Pass tokenized text through DistilBERT; extract the [CLS] token
        # embedding (index 0) which serves as a sentence-level representation
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_output.last_hidden_state[:, 0, :]  # shape: (batch, 768)
        x_text = self.text_dense(cls_output)                 # shape: (batch, DENSE_UNITS)

        # --- Structured branch ---
        # Process tabular features through the two-layer MLP
        x_struct = self.struct_branch(structured)  # shape: (batch, DENSE_UNITS)

        # --- Merge ---
        # Concatenate text and structured representations along the feature axis
        merged = torch.cat([x_text, x_struct], dim=1)  # shape: (batch, DENSE_UNITS*2)
        x = self.shared(merged)                         # shape: (batch, DENSE_UNITS)

        # --- Output heads ---
        # Produce raw logits for each classification task
        # (CrossEntropyLoss applies softmax internally, so no activation here)
        cat_logits = self.category_head(x)
        sub_logits = self.subcategory_head(x)

        return cat_logits, sub_logits