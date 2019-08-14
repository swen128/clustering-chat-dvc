import argparse

import torch
from pytorch_transformers.modeling_bert import BertConfig, BertForPreTraining, load_tf_weights_in_bert


def main(tf_checkpoint_path: str, bert_config_file: str, pytorch_dump_path: str):
    config = BertConfig.from_json_file(bert_config_file)
    model = BertForPreTraining(config)

    load_tf_weights_in_bert(model, config, tf_checkpoint_path)
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tf_checkpoint_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to the TensorFlow checkpoint path.")
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--pytorch_dump_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to the output PyTorch model.")
    args = parser.parse_args()

    main(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)
