import os
import argparse

from pydub.playback import play


def play(file_path: str) -> None:
    os.system(f'{file_path}')


def demo(arg_input: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str)
    args = parser.parse_args()

    play(f'demo\\{args.input}')
