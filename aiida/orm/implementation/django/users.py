# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""Django user module"""

import functools

from aiida.backends.djsite.db import models
from aiida.backends.djsite.db.models import DbUser
from aiida.orm.implementation.users import BackendUser, BackendUserCollection

from . import entities, utils

__all__ = ('DjangoUser', 'DjangoUserCollection')


class DjangoUser(entities.DjangoModelEntity[models.DbUser], BackendUser):
    """The Django user class"""

    MODEL_CLASS = models.DbUser

    def __init__(self, backend, email, first_name, last_name, institution):
        # pylint: disable=too-many-arguments
        super().__init__(backend)
        self._dbmodel = utils.ModelWrapper(
            DbUser(email=email, first_name=first_name, last_name=last_name, institution=institution)
        )

    @property
    def email(self):
        return self._dbmodel.email

    @email.setter
    def email(self, email):
        self._dbmodel.email = email

    @property
    def first_name(self):
        return self._dbmodel.first_name

    @first_name.setter
    def first_name(self, first_name):
        self._dbmodel.first_name = first_name

    @property
    def last_name(self):
        return self._dbmodel.last_name

    @last_name.setter
    def last_name(self, last_name):
        self._dbmodel.last_name = last_name

    @property
    def institution(self):
        return self._dbmodel.institution

    @institution.setter
    def institution(self, institution):
        self._dbmodel.institution = institution


class DjangoUserCollection(BackendUserCollection):
    """The Django collection of users"""

    ENTITY_CLASS = DjangoUser

    def create(self, email, first_name='', last_name='', institution=''):  # pylint: disable=arguments-differ
        """
        Create a user with the provided email address

        :return: A new user object
        :rtype: :class:`aiida.orm.implementation.django.users.DjangoUser`
        """
        # pylint: disable=abstract-class-instantiated
        return DjangoUser(self.backend, email, first_name, last_name, institution)

    def find(self, email=None, id=None):  # pylint: disable=redefined-builtin, invalid-name
        """
        Find users in this collection

        :param email: optional email address filter
        :param id: optional id filter
        :return: a list of the found users
        :rtype: list
        """
        # Constructing the default query
        import operator

        from django.db.models import Q  # pylint: disable=import-error, no-name-in-module
        query_list = []

        # If an id is specified then we add it to the query
        if id is not None:
            query_list.append(Q(pk=id))

        # If an email is specified then we add it to the query
        if email is not None:
            query_list.append(Q(email=email))

        if not query_list:
            dbusers = DbUser.objects.all()
        else:
            dbusers = DbUser.objects.filter(functools.reduce(operator.and_, query_list))
        found_users = []
        for dbuser in dbusers:
            found_users.append(self.from_dbmodel(dbuser))
        return found_users
