###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the code, visit http://www.aiida.net         #
#                                                                         #
# This code is licensed under the MIT license.                            #
# See the top-level LICENSE file for license details.                     #
###########################################################################
"""Tests for recovery from PostgreSQL connection failures."""

import asyncio

import plumpy.persistence
import psycopg
import pytest
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import object_session

from aiida import orm
from aiida.engine.processes.launcher import ProcessLauncher


@pytest.mark.requires_psql
def test_continuation_lookup_disconnect_does_not_detach_other_entities(backend):
    """A disconnect in the launcher's initial process lookup should not detach unrelated entities."""
    user = orm.User('before@example.org').store()
    process_node = orm.WorkChainNode().store()
    session = backend.get_session()
    engine = session.get_bind()
    config = backend.profile.storage_config

    def fail_on_savepoint(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith('SAVEPOINT'):
            backend_pid = conn.connection.driver_connection.info.backend_pid
            with psycopg.connect(
                host=config['database_hostname'],
                port=config['database_port'],
                dbname=config['database_name'],
                user=config['database_username'],
                password=config['database_password'],
                autocommit=True,
            ) as admin_conn:
                assert admin_conn.execute('SELECT pg_terminate_backend(%s)', (backend_pid,)).fetchone()[0]

    event.listen(engine, 'before_cursor_execute', fail_on_savepoint)
    loop = asyncio.new_event_loop()
    launcher = ProcessLauncher(loop=loop, load_context=plumpy.persistence.LoadSaveContext())
    try:
        with pytest.raises(OperationalError) as exception:
            loop.run_until_complete(launcher._continue(None, process_node.pk, nowait=False))
    finally:
        loop.close()
        event.remove(engine, 'before_cursor_execute', fail_on_savepoint)

    assert exception.value.connection_invalidated
    assert object_session(user._backend_entity.model._model) is session
    assert user.email == 'before@example.org'
    assert process_node.process_state is None


@pytest.mark.requires_psql
def test_query_error_is_preserved_when_rollback_and_logging_fail(backend, monkeypatch):
    """Failures during cleanup must not mask the original QueryBuilder exception."""
    from aiida.storage.psql_dos.orm.querybuilder import main

    session = backend.get_session()
    original_error = RuntimeError('query failed')

    def fail_query(*args, **kwargs):
        raise original_error

    def fail_rollback():
        raise RuntimeError('rollback failed')

    def fail_log(*args, **kwargs):
        raise RuntimeError('database log handler failed')

    monkeypatch.setattr(session, 'execute', fail_query)
    monkeypatch.setattr(session, 'rollback', fail_rollback)
    monkeypatch.setattr(main.LOGGER, 'exception', fail_log)

    with pytest.raises(RuntimeError, match='query failed') as exception:
        orm.QueryBuilder().append(orm.User).all()

    assert exception.value is original_error


def test_storage_release_continues_after_session_removal_failure():
    """Release should preserve its factory for retry when session removal fails during rollback."""
    from unittest.mock import Mock

    from aiida.storage.psql_dos.backend import _PsqlDosResources

    original = RuntimeError('session cleanup failed')
    factory = Mock()
    engine = Mock()
    factory.bind = engine
    factory.remove.side_effect = [original, None]
    resources = _PsqlDosResources()
    resources.session_factory = factory

    with pytest.raises(RuntimeError) as raised:
        resources.release()

    assert raised.value is original
    factory.expunge_all.assert_called_once_with()
    factory.remove.assert_called_once_with()
    engine.dispose.assert_called_once_with()
    assert resources.session_factory is factory

    resources.release()
    assert factory.remove.call_count == 2
    assert engine.dispose.call_count == 2
    assert resources.session_factory is None
