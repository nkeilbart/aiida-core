# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
# pylint: disable=invalid-name
"""Database migration."""
from django.db import migrations, models

from aiida.backends.djsite.db.migrations import upgrade_schema_version

REVISION = '1.0.13'
DOWN_REVISION = '1.0.12'


class Migration(migrations.Migration):
    """Database migration."""

    dependencies = [
        ('db', '0012_drop_dblock'),
    ]

    # An amalgamation from django:django/contrib/auth/migrations/
    # these changes are already the default for SQLA at this point
    operations = [
        migrations.AlterField(
            model_name='dbuser',
            name='last_login',
            field=models.DateTimeField(null=True, verbose_name='last login', blank=True),
        ),
        migrations.AlterField(
            model_name='dbuser',
            name='email',
            field=models.EmailField(max_length=254, verbose_name='email address', blank=True),
        ),
        upgrade_schema_version(REVISION, DOWN_REVISION)
    ]
