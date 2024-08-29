import torch
import pytorchvideo.data
from tokenizers import Tokenizer
from video2text import get_model, vivit_model
from torch.utils.data import DataLoader

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    Resize,
)


'''
Initialize of the model with trained weights and define helper functions.
'''


def get_config():
    return {
        "seq_len": 15,
        "d_model": 768,
        "lang_tgt": "ne",
    }


config = get_config()


target_tokenizer = Tokenizer.from_file(
    str('tokenizer_sign_lang_ne.json'))


# initialize the model
v2t_model = get_model(config=config, enc_model=vivit_model,
                      vocab_tgt_len=target_tokenizer.get_vocab_size())


# loading the saved model
model_path = '44_mtrain.pt'
lm = torch.load(model_path, map_location='cpu')

# load the trained parameter into the model
v2t_model.load_state_dict(lm['model_state_dict'])

v2t_model.to(torch.device('cpu'))


resize_to = (224, 224)
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    Resize(resize_to),
                    UniformTemporalSubsample(60),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                ]
            ),
        ),

    ]
)


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def greedy_decode(model, src_video, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(src_video=src_video)
#     encoder_output = (torch.randint(2,7,(1,784,768))).type_as(encoder_output).to(device)

#     print(f'encoder_output: {encoder_output[:,392:400,:20]}')
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(
        src_video.type(torch.LongTensor)).to(device)

#     print(f"decoder input: {decoder_input,decoder_input.shape}")
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(
            1)).type_as(src_video.type(torch.LongTensor)).to(device)

        # calculate output
        out = model.decode(encoder_output=encoder_output,
                           src_mask=None, tgt=decoder_input, tgt_mask=decoder_mask)

        # get next token
        prob = model.project(out[:, -1])

        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(src_video.type(torch.LongTensor)).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def beam_search_decode(model, beam_size, src_video, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(src_video=src_video)
    # Initialize the decoder input with the sos token
    decoder_initial_input = torch.empty(1, 1).fill_(
        sos_idx).type_as(src_video.type(torch.LongTensor)).to(device)

    # Create a candidate list
    candidates = [(decoder_initial_input, 1)]

    while True:

        # If a candidate has reached the maximum length, it means we have run the decoding for at least max_len iterations, so stop the search
        if any([cand.size(1) == max_len for cand, _ in candidates]):
            break

        # Create a new list of candidates
        new_candidates = []

        for candidate, score in candidates:

            # Do not expand candidates that have reached the eos token
            if candidate[0][-1].item() == eos_idx:
                continue

            # Build the candidate's mask
            candidate_mask = causal_mask(candidate.size(1)).type_as(
                src_video.type(torch.LongTensor)).to(device)

            # calculate output
            out = model.decode(encoder_output=encoder_output,
                               src_mask=None, tgt=candidate, tgt_mask=candidate_mask)

            # get next token probabilities
            prob = model.project(out[:, -1])

            # get the top k candidates
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)

            for i in range(beam_size):
                # for each of the top k candidates, get the token and its probability
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                token_prob = topk_prob[0][i].item()
                # create a new candidate by appending the token to the current candidate
                new_candidate = torch.cat([candidate, token], dim=1)
                # We sum the log probabilities because the probabilities are in log space
                new_candidates.append((new_candidate, score + token_prob))

        # Sort the new candidates by their score
        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        # Keep only the top k candidates
        candidates = candidates[:beam_size]

        # If all the candidates have reached the eos token, stop
        if all([cand[0][-1].item() == eos_idx for cand, _ in candidates]):
            break

    # Return the best candidate
    return candidates[0][0].squeeze()


def get_video_tensor(video_path):
    labeled_video_paths = [
        (video_path, {'label': 2})]
    print(video_path)

    infer_dataset = pytorchvideo.data.LabeledVideoDataset(
        labeled_video_paths=labeled_video_paths,
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", 10),
        decode_audio=False,
        transform=val_transform,
    )

    loader = DataLoader(infer_dataset, batch_size=1)
    vid = next(iter(loader))['video']
    return vid


def run_inference(model, video, tokenizer_tgt, max_len, device):

    model.eval()
    with torch.no_grad():

        # (b, channels, numframes, height, width)
        encoder_input = video['video'].to(device)

        # check that the batch size is 1
        assert encoder_input.size(
            0) == 1, "Batch size must be 1 for validation"

        # model_out = greedy_decode(
        #     model, encoder_input, None, tokenizer_tgt, max_len, device)

        model_out_beam = beam_search_decode(
            model, 2, encoder_input, None, tokenizer_tgt, max_len, device)

        model_out_text = tokenizer_tgt.decode(
            model_out_beam.detach().cpu().numpy())

        return {'translation': model_out_text}
