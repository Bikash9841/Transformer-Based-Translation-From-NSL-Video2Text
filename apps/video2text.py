from transformers import VivitImageProcessor, VivitModel
import torch
from decoder import Decoder, DecoderBlock, InputEmbeddings, PositionalEncoding, MultiHeadAttentionBlock, ResidualConnection, ProjectionLayer, FeedForwardBlock

model_checkpoint = "google/vivit-b-16x2-kinetics400"
image_processor = VivitImageProcessor.from_pretrained(model_checkpoint)

vivit_model = VivitModel.from_pretrained(model_checkpoint)


# To remove the pooler layer from the model, outputs what it gets from the previous layer
class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# patch size of 32*32
vivit_model.config.tubelet_size = [2, 32, 32]

# 6 encoder block stacks
vivit_model.config.num_hidden_layers = 12

# dropout set to 0.1
vivit_model.config.hidden_dropout_prob = 0.3

# number of frames extracting from each video
vivit_model.config.num_frames = 60

vivit_model = VivitModel(vivit_model.config)

vivit_model.pooler = Identity()


# Complete Video to Text model architecture
class Video2Text(torch.nn.Module):

    def __init__(self, encoder, decoder: Decoder, tgt_embed: InputEmbeddings, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.video_encoder = encoder
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src_video):
        # (batch,num_frames, num_channels, height, width)
        if src_video != None:
            perumuted_sample_test_video = src_video.permute(0, 2, 1, 3, 4)

            inputs = {
                "pixel_values": perumuted_sample_test_video,
            }
            # forward pass
            outputs = self.video_encoder(**inputs)

#           first token in the sequence is the class token. so, we dont need that. (batchsize, seq_len, embedding)
            return outputs.last_hidden_state[:, 1:, :]
        else:
            return None

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def build_transformer(encoder_model, tgt_vocab_size: int, tgt_seq_len: int, d_model: int = 768, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Video2Text:

    # Create the embedding layers
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(
            d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(
            d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block,
                                     decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    video_encoder = encoder_model
    decoder = Decoder(d_model, torch.nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Video2Text(
        encoder=video_encoder, decoder=decoder, tgt_embed=tgt_embed, tgt_pos=tgt_pos, projection_layer=projection_layer)

    # Initialize the parameters
#     for p in transformer.decoder.parameters():
#         if p.dim() > 1:
#             torch.nn.init.xavier_uniform_(p)

    return transformer


def get_model(config, enc_model, vocab_tgt_len):
    v2t_model = build_transformer(encoder_model=enc_model, tgt_vocab_size=vocab_tgt_len,
                                  tgt_seq_len=config['seq_len'], d_model=config['d_model'])
    return v2t_model
