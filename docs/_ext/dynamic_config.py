import os

from datetime import datetime
from importlib.metadata import version as get_version
from packaging.version import Version


dated_copyright = f"{datetime.now().strftime('%Y, Mira Geoscience Ltd')}"

def get_copyright_notice():
    return f"Copyright {dated_copyright}."


def setup(app):
    # The full version, including alpha/beta/rc tags.
    release = get_version("geoh5py")
    # The shorter X.Y.Z version.
    version = Version(release).base_version

    app.config.project_copyright = f"{dated_copyright}"
    app.config.release = release
    app.config.version = version
    app.config.html_theme_options = {
        "description": f"version {release}",
        "fixed_sidebar": True,
        "logo_name": "Geoh5py",
        "show_relbars": True,
    }
    app.config.rst_epilog = f"\n.. |copyright_notice| replace:: {get_copyright_notice()}\n"

    app.googleanalytics_id = os.environ.get("GOOGLE_ANALYTICS_ID", "")
    if not app.googleanalytics_id:
        print("DEBUG: GOOGLE_ANALYTICS_ID is not set")
        app.googleanalytics_enabled = False
    else:
        print(f"DEBUG: GOOGLE_ANALYTICS_ID set to: {app.googleanalytics_id}")

    return {"version": "0.1", "parallel_read_safe": True}
