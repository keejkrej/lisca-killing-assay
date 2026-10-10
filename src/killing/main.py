from killing import commands  # noqa: F401
from killing.app import app


def run() -> None:
    app()


if __name__ == "__main__":
    run()
