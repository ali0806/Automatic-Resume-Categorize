import torch.nn as nn
import torch

class BertResumeClassifier(nn.Module):
    """
    A BERT-based neural network model for resume classification.

    Args:
        bert_model (transformers.BertModel): Pretrained BERT model.
        num_classes (int): Number of output classes.
        dropout_prob (float): Dropout probability for regularization.
        l2_reg (float): L2 regularization factor.

    Attributes:
        bert (transformers.BertModel): Pretrained BERT model.
        intermediate_layer (nn.Linear): Linear layer for intermediate feature transformation.
        dropout (nn.Dropout): Dropout layer for regularization.
        output_layer (nn.Linear): Final linear layer for output logits.
    """

    def __init__(self, bert_model, num_classes, dropout_prob=0.3, l2_reg=1e-5):
        super(BertResumeClassifier, self).__init__()
        self.bert = bert_model
        self.intermediate_layer = nn.Linear(768, 512)
        self.dropout = nn.Dropout(dropout_prob)
        self.output_layer = nn.Linear(512, num_classes)

        # Initialize weights
        self.intermediate_layer.weight.data = nn.init.kaiming_normal_(self.intermediate_layer.weight.data)
        self.intermediate_layer.bias.data.fill_(0)
        self.output_layer.weight.data = nn.init.kaiming_normal_(self.output_layer.weight.data)
        self.output_layer.bias.data.fill_(0)

        self.l2_reg = l2_reg

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs.
            attention_mask (torch.Tensor): Tensor of attention masks.

        Returns:
            torch.Tensor: Logits output by the model.
        """
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)[1]
        intermediate = self.intermediate_layer(bert_outputs)
        intermediate = self.dropout(intermediate)
        logits = self.output_layer(intermediate)
        return logits
