import torch


def inputs_data() -> torch.Tensor:
    inputs = [
        [0.43, 0.15, 0.89],  # your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],  # step
    ]
    inputs = torch.tensor(inputs)

    return inputs


def main_single_self_attention():
    inputs = inputs_data()

    # Task: apply the self-attention to 2nd element ("journey") and find the context vector

    # step - 1: find the attention score for 2nd element by taking the dot product with each input
    attention_scores = torch.empty(inputs.shape[0])
    for ix, vector in enumerate(inputs):
        attention_scores[ix] = torch.dot(vector, inputs[1])
    print(f'Attention Scores: {attention_scores}')

    # step - 2: normalize the attention scores using softmax to get attention weights
    attention_weights = torch.nn.functional.softmax(attention_scores, dim=0)
    print(f'Attention Weights: {attention_weights}')
    print(f'Sum of attention weights: {attention_weights.sum()}')

    # step - 3: find the context vector by multiplying each input with the corresponding attention weight
    # and summing them up.
    context_vector = torch.zeros(inputs.shape[1])
    for ix, el in enumerate(inputs):
        context_vector += el * attention_weights[ix].item()
    print(f'Context Vector: {context_vector}')


if __name__ == '__main__':
    print('--- Single Self Attention ---')
    main_single_self_attention()
    print()
