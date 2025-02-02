# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""Provides import functionalities."""
from aiida.tools.importexport.dbimport.utils import IMPORT_LOGGER

__all__ = ('import_data', 'IMPORT_LOGGER')


def import_data(in_path, group=None, **kwargs):
    """Import exported AiiDA archive to the AiiDA database and repository.

    Proxy function for the backend-specific import functions.
    If ``in_path`` is a folder, calls extract_tree; otherwise, tries to detect the compression format
    (zip, tar.gz, tar.bz2, ...) and calls the correct function.

    Note, the logging level and progress reporter should be set externally, for example::

        from aiida.common.progress_reporter import set_progress_bar_tqdm

        IMPORT_LOGGER.setLevel('DEBUG')
        set_progress_bar_tqdm(leave=True)
        import_data(...)

    :param in_path: the path to a file or folder that can be imported in AiiDA.
    :type in_path: str

    :param group: Group wherein all imported Nodes will be placed.
    :type group: :py:class:`~aiida.orm.groups.Group`

    :param extras_mode_existing: 3 letter code that will identify what to do with the extras import.
        The first letter acts on extras that are present in the original node and not present in the imported node.
        Can be either:
        'k' (keep it) or
        'n' (do not keep it).
        The second letter acts on the imported extras that are not present in the original node.
        Can be either:
        'c' (create it) or
        'n' (do not create it).
        The third letter defines what to do in case of a name collision.
        Can be either:
        'l' (leave the old value),
        'u' (update with a new value),
        'd' (delete the extra), or
        'a' (ask what to do if the content is different).
    :type extras_mode_existing: str

    :param extras_mode_new: 'import' to import extras of new nodes or 'none' to ignore them
    :type extras_mode_new: str

    :param comment_mode: Comment import modes (when same UUIDs are found).
        Can be either:
        'newest' (will keep the Comment with the most recent modification time (mtime)) or
        'overwrite' (will overwrite existing Comments with the ones from the import file).
    :type comment_mode: str

    :return: New and existing Nodes and Links.
    :rtype: dict

    :raises `~aiida.tools.importexport.common.exceptions.ArchiveImportError`: if there are any internal errors when
        importing.
    """
    from aiida.backends import BACKEND_DJANGO, BACKEND_SQLA
    from aiida.manage import configuration
    from aiida.tools.importexport.common.exceptions import ArchiveImportError

    backend = configuration.PROFILE.database_backend

    if backend == BACKEND_SQLA:
        from aiida.tools.importexport.dbimport.backends.sqla import import_data_sqla
        IMPORT_LOGGER.debug('Calling import function import_data_sqla for the %s backend.', backend)
        return import_data_sqla(in_path, group=group, **kwargs)

    if backend == BACKEND_DJANGO:
        from aiida.tools.importexport.dbimport.backends.django import import_data_dj
        IMPORT_LOGGER.debug('Calling import function import_data_dj for the %s backend.', backend)
        return import_data_dj(in_path, group=group, **kwargs)

    # else
    raise ArchiveImportError(f'Unknown backend: {backend}')
