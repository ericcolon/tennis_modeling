import os


REPO_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir,
        os.pardir,
        os.pardir,
        os.pardir,
        os.pardir
    )
)
DATA_DIR = os.path.join(REPO_DIR, 'data')
