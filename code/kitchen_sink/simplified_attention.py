"""
Script to try manually coding a self-attention.
"""
import torch


def inputs_data() -> torch.Tensor:
    """
    Returns the input data.
    """
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
    """
    This main function finds the self-attention of the second element for illustration.
    """
    inputs = inputs_data()

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


def main_full():
    """
    This main function attempts to find the self-attention of the entire input.
    """
    inputs = inputs_data()

    # step - 1: find attention scores for all the inputs
    size = (
        inputs.shape[0],  # each row (input) will have one vector of attention score
        inputs.shape[0],  # each vector will have the number of attention scores as the number of inputs
    )
    attention_scores = torch.zeros(size)
    for query_ix, query in enumerate(inputs):
        # compute attention score for each input
        for key_ix, key in enumerate(inputs):
            # with each input
            attention_scores[query_ix][key_ix] = torch.dot(query, key)
    print('Attention Scores')
    print('===============')
    print(attention_scores)

    # step - 2: find attention weights for all the inputs
    attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)
    print('Attention Weights')
    print('================')
    print(attention_weights)

    # step - 3: find the context vector of all the inputs
    size = (
        inputs.shape[0],  # the number of inputs
        inputs.shape[1],  # the embedding size of each input
    )
    context_vector = torch.zeros(size)
    for context_ix, _ in enumerate(context_vector):
        for input_ix, el in enumerate(inputs):
            context_vector[context_ix] += el * attention_weights[context_ix][input_ix]
    print('Context Vectors')
    print('==============')
    print(context_vector)


def main_full_short_code():
    """
    This finds the full self-attention (like above main function), but with very short code.
    """
    inputs = inputs_data()

    # step - 1: find attention scores for all inputs
    attention_scores = inputs @ inputs.T

    # step - 2: find attention weights
    attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)

    # step - 3: find the context vector
    context_vector = attention_weights @ inputs
    print('Context Vectors')
    print('===============')
    print(context_vector)


if __name__ == '__main__':
    print('--- Single Self Attention ---')
    main_single_self_attention()
    print()

    print('--- Full Self Attention ---')
    main_full()
    print()

    print('--- Full Self Attention Short Code ---')
    main_full_short_code()
    print()
