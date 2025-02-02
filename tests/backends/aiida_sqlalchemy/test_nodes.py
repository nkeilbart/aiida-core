# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
# pylint: disable=import-error,no-name-in-module
"""Tests for nodes, attributes and links."""

from aiida import orm
from aiida.backends.testbase import AiidaTestCase
from aiida.orm import Data


class TestNodeBasicSQLA(AiidaTestCase):
    """These tests check the basic features of nodes(setting of attributes, copying of files, ...)."""

    def test_settings(self):
        """Test the settings table (similar to Attributes, but without the key."""
        from aiida.backends.sqlalchemy import get_scoped_session
        from aiida.backends.sqlalchemy.models.settings import DbSetting
        session = get_scoped_session()

        from pytz import UTC
        from sqlalchemy.exc import IntegrityError

        from aiida.common import timezone

        DbSetting.set_value(key='pippo', value=[1, 2, 3])

        # s_1 = DbSetting.objects.get(key='pippo')
        s_1 = DbSetting.query.filter_by(key='pippo').first()  # pylint: disable=no-member

        self.assertEqual(s_1.getvalue(), [1, 2, 3])

        s_2 = DbSetting(key='pippo')
        s_2.time = timezone.datetime.now(tz=UTC)
        with self.assertRaises(IntegrityError):
            with session.begin_nested():
                # same name...
                session.add(s_2)

        # Should replace pippo
        DbSetting.set_value(key='pippo', value='a')
        s_1 = DbSetting.query.filter_by(key='pippo').first()  # pylint: disable=no-member

        self.assertEqual(s_1.getvalue(), 'a')

    def test_load_nodes(self):
        """Test for load_node() function."""
        from aiida.backends.sqlalchemy import get_scoped_session
        from aiida.orm import load_node

        a_obj = Data()
        a_obj.store()

        self.assertEqual(a_obj.pk, load_node(identifier=a_obj.pk).pk)
        self.assertEqual(a_obj.pk, load_node(identifier=a_obj.uuid).pk)
        self.assertEqual(a_obj.pk, load_node(pk=a_obj.pk).pk)
        self.assertEqual(a_obj.pk, load_node(uuid=a_obj.uuid).pk)

        session = get_scoped_session()

        try:
            session.begin_nested()
            with self.assertRaises(ValueError):
                load_node(identifier=a_obj.pk, pk=a_obj.pk)
        finally:
            session.rollback()

        try:
            session.begin_nested()
            with self.assertRaises(ValueError):
                load_node(pk=a_obj.pk, uuid=a_obj.uuid)
        finally:
            session.rollback()

        try:
            session.begin_nested()
            with self.assertRaises(TypeError):
                load_node(pk=a_obj.uuid)
        finally:
            session.rollback()

        try:
            session.begin_nested()
            with self.assertRaises(TypeError):
                load_node(uuid=a_obj.pk)
        finally:
            session.rollback()

        try:
            session.begin_nested()
            with self.assertRaises(ValueError):
                load_node()
        finally:
            session.rollback()

    def test_multiple_node_creation(self):
        """
        This test checks that a node is not added automatically to the session
        (and subsequently committed) when a user is in the session.
        It tests the fix for the issue #234
        """
        import aiida.backends.sqlalchemy
        from aiida.backends.sqlalchemy.models.node import DbNode
        from aiida.common.utils import get_new_uuid

        backend = self.backend

        # Get the automatic user
        dbuser = backend.users.create(f'{self.id()}@aiida.net').store().dbmodel
        # Create a new node but don't add it to the session
        node_uuid = get_new_uuid()
        DbNode(user=dbuser, uuid=node_uuid, node_type=None)

        session = aiida.backends.sqlalchemy.get_scoped_session()

        # Query the session before commit
        res = session.query(DbNode.uuid).filter(DbNode.uuid == node_uuid).all()
        self.assertEqual(len(res), 0, 'There should not be any nodes with this UUID in the session/DB.')

        # Commit the transaction
        session.commit()

        # Check again that the node is not in the DB
        res = session.query(DbNode.uuid).filter(DbNode.uuid == node_uuid).all()
        self.assertEqual(len(res), 0, 'There should not be any nodes with this UUID in the session/DB.')

        # Get the automatic user
        dbuser = orm.User.objects.get_default().backend_entity.dbmodel
        # Create a new node but now add it to the session
        node_uuid = get_new_uuid()
        node = DbNode(user=dbuser, uuid=node_uuid, node_type=None)
        session.add(node)

        # Query the session before commit
        res = session.query(DbNode.uuid).filter(DbNode.uuid == node_uuid).all()
        self.assertEqual(len(res), 1, f'There should be a node in the session/DB with the UUID {node_uuid}')

        # Commit the transaction
        session.commit()

        # Check again that the node is in the db
        res = session.query(DbNode.uuid).filter(DbNode.uuid == node_uuid).all()
        self.assertEqual(len(res), 1, f'There should be a node in the session/DB with the UUID {node_uuid}')
