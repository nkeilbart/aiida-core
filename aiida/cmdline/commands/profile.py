# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida_core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""
This allows to manage profiles from command line.
"""
import sys
import click
from aiida.cmdline.baseclass import VerdiCommandWithSubcommands



valid_processes = ['verdi', 'daemon']


class Profile(VerdiCommandWithSubcommands):
    """
    List AiiDA profiles, and set the default profile.

    Allow to see the list of AiiDA profiles, and to set the default profile
    (the to be used by any verdi command when no '-p' option is given).
    """

    def __init__(self):
        """
        A dictionary with valid commands and functions to be called.
        """
        self.valid_subcommands = {
            'setdefault': (self.profile_setdefault,
                           self.complete_processes_profiles),
            'list': (self.profile_list, self.complete_none),
            'delete': (self.profile_delete, self.complete_processes_profiles),
        }

    def complete_processes_profiles(self, subargs_idx, subargs):
        from aiida.common.setup import get_profiles_list

        if subargs_idx == 1:
            return "\n".join(get_profiles_list())
        elif subargs_idx == 0:
            return "\n".join(valid_processes)
        else:
            return ""

    def profile_setdefault(self, *args):
        from aiida.common.setup import set_default_profile

        valid_processes_strlist = ", ".join("'{}'".format(pr) for pr in
            valid_processes)

        if len(args) != 2:
            print >> sys.stderr, ("You have to pass (only) two parameters "
                                  "after 'profile setdefault', the name of")
            print >> sys.stderr, ("the process ({}) and the "
                                  "profile to be set as the default.".format(
                    valid_processes_strlist))
            sys.exit(1)

        process = args[0]
        if process not in valid_processes:
            print >> sys.stderr, ("'{}' is not a valid process. Choose it from "
                                  "the following list: {}.".format(
                    process, valid_processes_strlist))
            sys.exit(1)

        profile = args[1]
        # set default DB profiles
        set_default_profile(process, profile, force_rewrite=True)


    def profile_list(self, *args):
        from aiida.common.setup import get_profiles_list, get_default_profile
        from aiida.backends import settings

        current_profile = settings.AIIDADB_PROFILE
        default_profile = get_default_profile(
                settings.CURRENT_AIIDADB_PROCESS)
        default_daemon_profile = get_default_profile("daemon")
        if current_profile is None:
            current_profile = default_profile

        use_colors = False
        if args:
            try:
                if len(args) != 1:
                    raise ValueError
                if args[0] != "--color":
                    raise ValueError
                use_colors = True
            except ValueError:
                print >> sys.stderr, ("You can pass only one further argument, "
                                      "--color, to show the results with colors")
                sys.exit(1)

        if current_profile is not None:
            # print >> sys.stderr, "### The '>' symbol indicates the current default user ###"
            pass
        else:
            print >> sys.stderr, "### No default profile configured yet, "\
                                 "run 'verdi install'! ###"

        for profile in get_profiles_list():
            color_id = 39  # Default foreground color
            if profile == current_profile:
                symbol = ">"
                color_id = 31
            else:
                symbol = "*"

            if profile == default_profile:
                color_id = 34
                default_str = ' (DEFAULT)'
            else:
                default_str = ''
            if profile == default_daemon_profile:
                default_str += ' (DAEMON PROFILE)'

            if use_colors:
                start_color = "\x1b[{}m".format(color_id)
                end_color = "\x1b[0m"
                bold_sequence = "\x1b[1;{}m".format(color_id)
                nobold_sequence = "\x1b[0;{}m".format(color_id)
            else:
                start_color = ""
                end_color = ""
                bold_sequence = ""
                nobold_sequence = ""

            print "{}{} {}{}{} {}{}".format(
                start_color, symbol,
                bold_sequence, profile, default_str, nobold_sequence, end_color)

    def profile_delete(self, *args):
        """ Deletes profile

        Asks whether to delete associated database and associated database
        user.
        """
        from aiida.cmdline.verdilib import Quicksetup
        from aiida.common.setup import get_or_create_config, update_config

        #TODO: the functions used below should be moved outside Quicksetup
        q = Quicksetup()
        pg_info = q._get_pg_access()
        pg_execute = pg_info['method']
        dbinfo = pg_info['dbinfo']

        confs = get_or_create_config()
        profiles = confs.get('profiles',{})

        profiles_to_delete = args
        for profile_to_delete in profiles_to_delete:
            try:
                profile = profiles[profile_to_delete]
            except KeyError:
                raise ValueError('Profile "{}" does not exist'.format(profile_to_delete))

            db = profile.get('AIIDADB_NAME', '')
            if click.confirm('Delete associated database {}?\nWARNING: All data will be lost.'.format(db)):
                print("Deleting database {}.".format(db))
                q._drop_db(db, pg_execute, **dbinfo)

            user = profile.get('AIIDADB_USER', '')
            if click.confirm('Delete database user {}?'.format(user)):
                print("Deleting user {}.".format(user))
                q._drop_dbuser(user, pg_execute, **dbinfo)

            print("Deleting configuration for profile {}.".format(profile_to_delete))
            del profiles[profile_to_delete]
            update_config(confs)


