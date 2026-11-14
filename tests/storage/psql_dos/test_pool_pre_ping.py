###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""Tests for PostgreSQL connection pool resilience."""

import psycopg
import pytest

from aiida.storage.psql_dos.utils import create_sqlalchemy_engine


@pytest.mark.requires_psql
def test_pool_pre_ping_replaces_stale_connection(backend):
    """An idle connection terminated by PostgreSQL is replaced on its next checkout."""
    config = backend.profile.storage_config.copy()
    config['pool_pre_ping'] = True
    engine = create_sqlalchemy_engine(config)

    try:
        with engine.connect() as connection:
            backend_pid = connection.exec_driver_sql('SELECT pg_backend_pid()').scalar_one()

        with psycopg.connect(
            host=config['database_hostname'],
            port=config['database_port'],
            dbname=config['database_name'],
            user=config['database_username'],
            password=config['database_password'],
            autocommit=True,
        ) as admin_conn:
            assert admin_conn.execute('SELECT pg_terminate_backend(%s)', (backend_pid,)).fetchone()[0]

        with engine.connect() as connection:
            assert connection.exec_driver_sql('SELECT 1').scalar_one() == 1
    finally:
        engine.dispose()
