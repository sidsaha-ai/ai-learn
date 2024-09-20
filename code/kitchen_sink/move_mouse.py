"""
Auto mouse move to keep Mac active!
"""
import time

import pyautogui


def main():
    """
    The main method to start execution.
    """
    while True:
        time.sleep(5)
        pyautogui.moveRel(10, 0, duration=0.5)
        pyautogui.moveRel(-10, 0, duration=0.5)


if __name__ == '__main__':
    main()
