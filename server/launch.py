#!/usr/bin/env python3
"""Xihe training/inference service entry point
"""
import sys
import fire
import importlib
import multiprocessing

entries = {
    'gen_data': {'module': 'datasets.xihe', 'func': 'gen_data'},
    'gen_fps_data': {'module': 'datasets.xihe', 'func': 'gen_fps_data'},
    'gen_pointar_data': {'module': 'datasets.pointar', 'func': 'gen_data'},
    'post_message': {'module': 'model.utils', 'func': 'post_message'},
    'train': {'module': 'model', 'func': 'train_xihenet'},
    'serve': {'module': 'service', 'func': 'start_service'},
    'merge_rec': {'module': 'evaluation.real_world_testing', 'func': 'merge_rec'},
}


class Launcher:
    def __dir__(self):
        return list(entries.keys())

    def __getattr__(self, key):
        if key not in entries.keys():
            return None

        m = importlib.import_module(entries[key]['module'])
        return getattr(m, entries[key]['func'])

    def __call__(self):
        print('🚀 Launch via the following commands:', ', '.join(entries.keys()))


if __name__ == "__main__":
    # hot patch for enabling multiprocessing in data generation
    if len(sys.argv) > 1 and sys.argv[1] in ['gen_data', 'gen_fps_data', 'gen_pointar_data']:
        multiprocessing.set_start_method("spawn")

    # I don't like the way that Fire uses help screen instead of
    # just printing messages to std out. So, patch it:
    fire.core.Display = lambda lines, out: print(*lines, file=out)

    fire.Fire(Launcher)
