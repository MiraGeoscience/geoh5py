from pathlib import Path

import geoh5py
from poetry_publish.publish import poetry_publish


def publish():
    poetry_publish(
        package_root=Path(geoh5py.__file__).parent.parent,
        version=geoh5py.__version__,
    )
