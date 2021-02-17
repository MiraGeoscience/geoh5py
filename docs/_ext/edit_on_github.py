"""
Sphinx extension to add ReadTheDocs-style "Edit on GitHub" links to the
sidebar.

Loosely based on https://github.com/astropy/astropy/pull/347
"""

#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.

import os
import warnings

__licence__ = "BSD (3 clause)"


def get_github_url(app, view, path):
    if app.config.edit_on_github_directory is not None:
        return "https://github.com/{project}/{view}/{branch}/{dir}/{path}".format(
            project=app.config.edit_on_github_project,
            view=view,
            branch=app.config.edit_on_github_branch,
            dir=app.config.edit_on_github_directory,
            path=path,
        )
    else:
        return "https://github.com/{project}/{view}/{branch}/{path}".format(
            project=app.config.edit_on_github_project,
            view=view,
            branch=app.config.edit_on_github_branch,
            path=path,
        )


def html_page_context(app, pagename, templatename, context, doctree):
    if templatename != "page.html":
        return

    if not app.config.edit_on_github_project:
        warnings.warn("edit_on_github_project not specified")
        return

    path = os.path.relpath(doctree.get("source"), app.builder.srcdir)
    show_url = get_github_url(app, "blob", path)
    edit_url = get_github_url(app, "edit", path)

    # context['show_on_github_url'] = show_url
    context["edit_on_github_url"] = edit_url


def setup(app):
    app.add_config_value("edit_on_github_project", "", True)
    app.add_config_value("edit_on_github_branch", "master", True)
    app.add_config_value("edit_on_github_directory", None, True)
    app.connect("html-page-context", html_page_context)
