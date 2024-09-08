"""
This plots various common things needed.
"""

import matplotlib.pyplot as plt

class Plotter:

    @classmethod
    def plot_losses(cls, losses: list[dict]) -> None:
        """
        Plot the input list of losses values.
        """
        loss = [entry['loss'] for entry in losses]
        lr = [entry['lr'] for entry in losses]

        unique_lrs = []
        segments = []
        current_lr = lr[0]
        segment_start = 0
        for ix, lr in enumerate(lr):
            if lr != current_lr:
                unique_lrs.append(current_lr)
                segments.append((segment_start, ix))
                current_lr = lr
                segment_start = ix
        unique_lrs.append(current_lr)
        segments.append((segment_start, len(losses)))

        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']

        plt.figure(figsize=(10, 6))
        for ix, (start, end) in enumerate(segments):
            color = colors[ix % len(colors)]
            plt.plot(
                range(start + 1, end + 1), loss[start:end], label=f'LR {unique_lrs[ix]}', color=color,
            )
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Progression')
        plt.legend()
        plt.tight_layout()
        plt.show()
