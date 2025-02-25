from transformers import PretrainedConfig, PreTrainedModel, AutoModel
import torch
import os

class BiEncoderConfig(PretrainedConfig):
    """
    Configuration class for the BiEncoder model.

    Attributes:
        model_type (str): The type of model (bi-encoder).
        encoder_name (str): The name of the encoder model.
        num_classes (int): The number of classes for classification.
        version (str): The version of the model.
    """

    model_type = "bi-encoder"

    def __init__(self, encoder_name="sentence-transformers/all-distilroberta-v1", num_classes=4, version="latest", **kwargs):
        """
        Initializes the BiEncoderConfig.

        Args:
            encoder_name (str): The name of the encoder model.
            num_classes (int): The number of classes for classification.
            version (str): The version of the model.
            **kwargs: Additional keyword arguments for PretrainedConfig.
        """
        super().__init__(**kwargs)
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.version = version

class BiEncoderWithClassifier(PreTrainedModel):
    """
    Bi-encoder model with a classifier on top.

    Attributes:
        config_class: The configuration class associated with the model.
        encoder (PreTrainedModel): The encoder model.
        classifier (torch.nn.Linear): The classifier layer.
    """

    config_class = BiEncoderConfig

    def __init__(self, config):
        """
        Initializes the BiEncoderWithClassifier.

        Args:
            config (BiEncoderConfig): Configuration object for the model.
        """
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass for the model.

        Args:
            input_ids (torch.Tensor): Input IDs for the encoder.
            attention_mask (torch.Tensor, optional): Attention mask for the encoder.
            token_type_ids (torch.Tensor, optional): Token type IDs for the encoder.

        Returns:
            torch.Tensor: Logits from the classifier.
        """
        outputs = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]  # Assuming the encoder returns (sequence_output, pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def save_pretrained(self, save_directory, **kwargs):
        """
        Saves the model and its components to the specified directory.

        Args:
            save_directory (str): Directory to save the model.
            **kwargs: Additional keyword arguments for the saving process.
        """
        super().save_pretrained(save_directory, **kwargs)
        self.encoder.save_pretrained(save_directory)
        torch.save(self.classifier.state_dict(), os.path.join(save_directory, f"classifier_{self.config.version}.pt"))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Loads the model from a pretrained checkpoint.

        Args:
            pretrained_model_name_or_path (str): Name or path to the pretrained model.
            *model_args: Additional arguments for loading the model.
            **kwargs: Additional keyword arguments, including config.

        Returns:
            BiEncoderWithClassifier: An instance of the BiEncoderWithClassifier.
        """
        config = kwargs.pop("config", None)
        if config is None:
            config = BiEncoderConfig.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        model = cls(config)
        model.encoder = AutoModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        # Load the classifier with the specified version
        classifier_path = os.path.join(pretrained_model_name_or_path, f"classifier_{config.version}.pt")
        if os.path.exists(classifier_path):
            model.classifier.load_state_dict(torch.load(classifier_path))
        else:
            raise ValueError(f"Classifier weights for version {config.version} not found.")

        return model

if __name__ == "__main__":
    # test
    model = BiEncoderWithClassifier()
    queries = ["wireless headphones", "gaming laptop"]
    products = ["Bluetooth over-ear headphones", "High-performance gaming laptop"]

    logits = model(queries, products)
    print("\nLogits for Query-Product Pairs:\n", logits)