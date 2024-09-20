import pyautogui
import time

def main():
    while True:
        time.sleep(5)
        pyautogui.moveRel(10, 0, duration=0.5)
        pyautogui.moveRel(-10, 0, duration=0.5)


if __name__ == '__main__':
    main()
