"""
This contains the solution for the exercise https://tensorgym.com/exercises/5
"""
import torch

def solution(student_answers: torch.Tensor, question_points: torch.Tensor) -> torch.Tensor:
    """
    The implemnentation of the exercise solution.
    """
    scores = student_answers @ question_points
    top_student = torch.argmax(scores)

    return top_student


if __name__ == '__main__':
    cases = []
    # test case 1
    inputs_answers = [
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
    ]
    inputs_points = [2, 3, 1]
    outputs = 2
    cases.append({
        'inputs_answers': torch.tensor(inputs_answers),
        'inputs_points': torch.tensor(inputs_points),
        'outputs': torch.tensor(outputs),
    })

    # test case 2
    inputs_answers = [
        [1, 0],
        [0, 1],
        [1, 1],
    ]
    inputs_points = [2, 3]
    outputs = 2
    cases.append({
        'inputs_answers': torch.tensor(inputs_answers),
        'inputs_points': torch.tensor(inputs_points),
        'outputs': torch.tensor(outputs),
    })
    
    for ix, c in enumerate(cases):
        res = solution(c.get('inputs_answers'), c.get('inputs_points'))
        message = f'PASS: test case {ix + 1}' if torch.equal(res, c.get('outputs')) else f'FAIL: test case {ix + 1}'
        print(message)