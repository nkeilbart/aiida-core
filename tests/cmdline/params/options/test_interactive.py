# -*- coding: utf-8 -*-
###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""Unit tests for the InteractiveOption."""
import unittest

import click
from click.testing import CliRunner
from click.types import IntParamType

from aiida.cmdline.params.options import NON_INTERACTIVE
from aiida.cmdline.params.options.interactive import InteractiveOption


class Only42IntParamType(IntParamType):
    """
    Param type that only accepts 42 as valid value
    """
    name = 'only42int'

    def convert(self, value, param, ctx):
        newval = super().convert(value, param, ctx)
        if newval != 42:
            self.fail('Type validation: invalid, should be 42')
        return newval

    def __repr__(self):
        return 'ONLY42INT'


def validate_positive_number(ctx, param, value):  # pylint: disable=unused-argument
    """Validate that the number passed to this parameter is a positive number.

    :param ctx: the `click.Context`
    :param param: the parameter
    :param value: the value passed for the parameter
    :raises `click.BadParameter`: if the value is not a positive number
    """
    if not isinstance(value, (int, float)) or value < 0:
        from click import BadParameter
        raise BadParameter(f'{value} is not a valid positive number')


def validate_positive_number_with_echo(ctx, param, value):  # pylint: disable=unused-argument
    """Validate that the number passed to this parameter is a positive number.
       Also echos a message to the terminal

    :param ctx: the `click.Context`
    :param param: the parameter
    :param value: the value passed for the parameter
    :raises `click.BadParameter`: if the value is not a positive number
    """
    click.echo(f'Validating {value}')
    if not isinstance(value, (int, float)) or value < 0:
        from click import BadParameter
        raise BadParameter(f'{value} is not a valid positive number')


class InteractiveOptionTest(unittest.TestCase):
    """Unit tests for InteractiveOption."""

    # pylint: disable=too-many-public-methods, missing-docstring

    def simple_command(self, **kwargs):
        """Return a simple command with one InteractiveOption, kwargs get relayed to the option."""

        # pylint: disable=no-self-use

        @click.command()
        @click.option('--opt', prompt='Opt', cls=InteractiveOption, **kwargs)
        @NON_INTERACTIVE()
        def cmd(opt, non_interactive):  # pylint: disable=unused-argument
            """test command for InteractiveOption"""
            click.echo(str(opt))

        return cmd

    @classmethod
    def setUpClass(cls):
        cls.runner = CliRunner()

    def prompt_output(self, cli_input, converted=None):
        """Return expected output of simple_command, given a commandline cli_input string."""
        # pylint: disable=no-self-use

        return f'Opt: {cli_input}\n{converted or cli_input}\n'

    def test_callback_prompt_twice(self):
        """
        scenario: using InteractiveOption with type=float and callback that tests for positive number
        behaviour: should fail everytime either type validation or callback validation fails
        """
        cmd = self.simple_command(type=float, callback=validate_positive_number)
        runner = CliRunner()
        result = runner.invoke(cmd, [], input='string\n-1\n-1\n1\n')
        expected_1 = 'is not a valid floating point value'
        expected_2 = 'is not a valid positive number'
        expected_3 = 'is not a valid positive number'
        expected_4 = '1.0'
        self.assertIsNone(result.exception)
        lines = result.output.split('\n')
        self.assertIn(expected_1, lines[3])
        self.assertIn(expected_2, lines[6])
        self.assertIn(expected_3, lines[9])
        self.assertIn(expected_4, lines[12])

    def test_callback_prompt_only_once(self):
        """
        scenario: using InteractiveOption with type=float and callback that echos an additional message
        behaviour: the callback should be called at most once per prompt
        """
        cmd = self.simple_command(type=float, callback=validate_positive_number_with_echo)
        runner = CliRunner()
        result = runner.invoke(cmd, [], input='string\n-1\n1\n')
        self.assertIsNone(result.exception)
        expected_1 = 'Error: string is not a valid floating point value'
        expected_2 = 'Validating -1.0'
        expected_3 = 'Error: -1.0 is not a valid positive number'
        expected_4 = 'Validating 1.0'
        expected_5 = '1.0'
        lines = result.output.split('\n')
        #The callback should be called once per prompt
        #where type conversion was successful
        self.assertEqual(expected_1, lines[3])
        self.assertEqual(expected_2, lines[6])
        self.assertEqual(expected_3, lines[7])
        self.assertEqual(expected_4, lines[10])
        self.assertEqual(expected_5, lines[11])

    def test_prompt_str(self):
        """
        scenario: using InteractiveOption with type=str
        behaviour: giving no option prompts, accepts a string
        """
        cmd = self.simple_command(type=str)
        runner = CliRunner()
        result = runner.invoke(cmd, [], input='TEST\n')
        expected = self.prompt_output('TEST')
        self.assertIsNone(result.exception)
        self.assertIn(expected, result.output)

    def test_prompt_empty_input(self):
        """
        scenario: using InteractiveOption with type=str and invoking without options
        behaviour: pressing enter on empty line at prompt repeats the prompt without a message
        """
        cmd = self.simple_command(type=str)
        runner = CliRunner()
        result = runner.invoke(cmd, [], input='\nTEST\n')
        expected = 'Opt: \nOpt: TEST\nTEST\n'
        self.assertIsNone(result.exception)
        self.assertIn(expected, result.output)

    def test_prompt_help_default(self):
        """
        scenario: using InteractiveOption with type=str and no help parameter and invoking without options
        behaviour: entering '?' leads to a default help message being printed and prompt repeated
        """
        cmd = self.simple_command(type=str)
        runner = CliRunner()
        result = runner.invoke(cmd, [], input='?\nTEST\n')
        expected_1 = 'Opt: ?\n'
        expected_2 = 'Expecting text\n'
        expected_3 = 'Opt: TEST\nTEST\n'
        self.assertIsNone(result.exception)
        self.assertIn(expected_1, result.output)
        self.assertIn(expected_2, result.output)
        self.assertIn(expected_3, result.output)

    def test_prompt_help_custom(self):
        """
        scenario: using InteractiveOption with type=str and help message and invoking without options
        behaviour: entering '?' leads to the given help message being printed and the prompt repeated
        """
        cmd = self.simple_command(type=str, help='Please enter some text')
        runner = CliRunner()
        result = runner.invoke(cmd, [], input='?\nTEST\n')
        expected_1 = 'Opt: ?\n'
        expected_2 = 'Please enter some text\n'
        expected_3 = 'Opt: TEST\nTEST\n'
        self.assertIsNone(result.exception)
        self.assertIn(expected_1, result.output)
        self.assertIn(expected_2, result.output)
        self.assertIn(expected_3, result.output)

    def test_prompt_simple(self):
        """
        scenario: using InteractiveOption with type=bool
        behaviour: giving no option prompts, accepts 'true'
        """
        params = [(bool, 'true', 'True'), (int, '98', '98'), (float, '3.14e-7', '3.14e-07')]
        for ptype, cli_input, output in params:
            cmd = self.simple_command(type=ptype, help='help msg')
            runner = CliRunner()
            result = runner.invoke(cmd, [], input=f'\n?\n{cli_input}\n')
            expected_1 = 'Opt: \nOpt: ?\n'
            expected_2 = 'help msg\n'
            expected_2 += self.prompt_output(cli_input, output)
            self.assertIsNone(result.exception)
            self.assertIn(expected_1, result.output)
            self.assertIn(expected_2, result.output)

    @staticmethod
    def strip_line(text):
        """returns text without the last line"""
        return text.rsplit('\n')[0]

    def test_prompt_complex(self):
        """
        scenario: using InteractiveOption with type=float
        behaviour: giving no option prompts, accepts 3.14e-7
        """
        params = [(click.File(), __file__), (click.Path(exists=True), __file__)]
        for ptype, cli_input in params:
            cmd = self.simple_command(type=ptype, help='help msg')
            runner = CliRunner()
            result = runner.invoke(cmd, [], input=f'\n?\n{cli_input}\n')
            expected_1 = 'Opt: \nOpt: ?\n'
            expected_2 = 'help msg\n'
            expected_2 += self.strip_line(self.prompt_output(cli_input))
            self.assertIsNone(result.exception)
            self.assertIn(expected_1, result.output)
            self.assertIn(expected_2, result.output)

    def test_default_value_prompt(self):
        """
        scenario: using InteractiveOption with a default value, invoke without options
        behaviour: prompt, showing the default value, take default on empty cli_input.
        """
        returns = []
        cmd = self.simple_command(default='default')
        result = self.runner.invoke(cmd, [], input='\n')
        returns.append(result)
        expected = 'Opt [default]: \ndefault\n'
        self.assertIsNone(result.exception)
        self.assertIn(expected, result.output)
        result = self.runner.invoke(cmd, [], input='TEST\n')
        returns.append(result)
        expected = 'Opt [default]: TEST\nTEST\n'
        self.assertIsNone(result.exception)
        self.assertIn(expected, result.output)
        return returns

    def test_default_value_empty_opt(self):
        """
        scenario: InteractiveOption with default value, invoke with empty option (--opt=)
        behaviour: accept empty string as input
        """
        cmd = self.simple_command(default='default')
        runner = CliRunner()
        result = runner.invoke(cmd, ['--opt='])
        expected = '\n'
        self.assertIsNone(result.exception)
        self.assertEqual(result.output, expected)

    def test_default_value_ignore_character(self):
        """
        scenario: InteractiveOption with default value, invoke with "ignore default character" `!`
        behaviour: return empty string '' for the value

        Note: It should *not* return None, since this is indistinguishable from the option not being prompted for.
        """
        cmd = self.simple_command(default='default')
        runner = CliRunner()

        # Check the interactive mode, by not specifying the input on the command line and then enter `!` at the prompt
        result = runner.invoke(cmd, [], input='!\n')
        expected = ''
        self.assertIsNone(result.exception)
        self.assertIn(expected, result.output.split('\n')[3])  # Fourth line should be parsed value printed to stdout

        # In the non-interactive mode the special character `!` should have no special meaning and be accepted as is
        result = runner.invoke(cmd, ['--opt=!'])
        expected = '!\n'
        self.assertIsNone(result.exception)
        self.assertIn(expected, result.output)

    def test_opt_given_valid(self):
        """
        scenario: InteractiveOption, invoked with a valid value on the cmdline
        behaviour: accept valid value
        """
        cmd = self.simple_command(type=int)
        runner = CliRunner()
        result = runner.invoke(cmd, ['--opt=4'])
        expected = '4\n'
        self.assertIsNone(result.exception)
        self.assertEqual(result.output, expected)

    def test_opt_given_invalid(self):
        """
        scenario: InteractiveOption, invoked with a valid value on the cmdline
        behaviour: accept valid value
        """
        cmd = self.simple_command(type=int)
        runner = CliRunner()
        result = runner.invoke(cmd, ['--opt=foo'])
        self.assertIsNotNone(result.exception)
        self.assertIn('Invalid value', result.output)

    def test_non_interactive(self):
        """
        scenario: InteractiveOption, invoked with only --non-interactive (and the option is required)
        behaviout: fail
        """
        cmd = self.simple_command(required=True)
        runner = CliRunner()
        result = runner.invoke(cmd, ['--non-interactive'])
        self.assertIsNotNone(result.exception)
        self.assertIn('Usage: ', result.output)
        self.assertIn('Missing option', result.output)

    def test_non_interactive_default(self):
        """
        scenario: InteractiveOption, invoked with only --non-interactive
        behaviour: success
        """
        cmd = self.simple_command(default='default')
        runner = CliRunner()
        result = runner.invoke(cmd, ['--non-interactive'])
        self.assertIsNone(result.exception)
        self.assertEqual(result.output, 'default\n')

    @staticmethod
    def user_callback(_ctx, param, value):
        """
        A fake user callback ued for testing.

        :param _ctx: The click context
        :param param: The parameter name
        :param value: The parameter value
        :return: The validated parameter
        """
        if not value:
            return -1

        if value != 42:
            raise click.BadParameter('invalid', param=param)

        return value

    def test_after_callback_valid(self):
        """
        scenario: InteractiveOption with a user callback
        action: invoke with valid value
        behaviour: user callback runs & succeeds
        """
        cmd = self.simple_command(callback=self.user_callback, type=int)
        result = self.runner.invoke(cmd, ['--opt=42'])
        self.assertIsNone(result.exception)
        self.assertEqual(result.output, '42\n')

    def test_after_callback_invalid(self):
        """
        scenario: InteractiveOption with a user callback
        action: invoke with invalid value of right type
        behaviour: user callback runs & succeeds
        """
        cmd = self.simple_command(callback=self.user_callback, type=int)
        result = self.runner.invoke(cmd, ['--opt=234234'])
        self.assertIsNotNone(result.exception)
        self.assertIn('Invalid value', result.output)
        self.assertIn('invalid', result.output)

    def test_after_callback_wrong_type(self):
        """
        scenario: InteractiveOption with a user callback
        action: invoke with invalid value of wrong type
        behaviour: user callback does not run
        """
        cmd = self.simple_command(callback=self.user_callback, type=int)
        result = self.runner.invoke(cmd, ['--opt=bla'])
        self.assertIsNotNone(result.exception)
        self.assertIn('Invalid value', result.output)
        self.assertIn('bla', result.output)

    def test_after_callback_empty(self):
        """
        scenario: InteractiveOption with a user callback
        action: invoke with invalid value of wrong type
        behaviour: user callback does not run
        """
        cmd = self.simple_command(callback=self.user_callback, type=int)
        result = self.runner.invoke(cmd, ['--opt='])
        self.assertIsNotNone(result.exception)
        self.assertIn('Invalid value', result.output)
        self.assertNotIn('empty', result.output)

    def test_after_validation_interactive(self):
        """
        Test that the type validation gets called on values entered at a prompt.

        Scenario:
            * InteractiveOption with custom type and prompt set
            * invoked without passing the options
            * on prompt: first enter an invalid value, then a valid one

        Behaviour:
            * Prompt for the value
            * reject invalid value, prompt again
            * accept valid value
        """
        cmd = self.simple_command(callback=self.user_callback, type=Only42IntParamType())
        result = self.runner.invoke(cmd, [], input='23\n42\n')
        self.assertIsNone(result.exception)
        self.assertIn('Opt: 23\n', result.output)
        self.assertIn('Type validation: invalid', result.output)
        self.assertIn('Opt: 42\n42\n', result.output)

    def test_after_callback_default_noninteractive(self):
        """
        Test that the callback gets called on the default, in line with click 6 behaviour.

        Scenario:
            * InteractiveOption with user callback and invalid default
            * invoke with no options and --non-interactive

        Behaviour:
            * the default value gets passed through the callback and rejected
        """
        # pylint: disable=invalid-name
        cmd = self.simple_command(callback=self.user_callback, type=int, default=23)
        result = self.runner.invoke(cmd, ['--non-interactive'])
        self.assertIsNotNone(result.exception)
        self.assertIn('Invalid value', result.output)

    def test_default_empty_empty_cli(self):
        """Test that default="" allows to pass an empty cli option."""
        cmd = self.simple_command(default='', type=str)
        result = self.runner.invoke(cmd, ['--opt='])
        self.assertIsNone(result.exception)
        self.assertEqual(result.output, '\n')

    def test_default_empty_prompt(self):
        """Test that default="" allows to pass an empty cli option."""
        cmd = self.simple_command(default='', type=str)
        result = self.runner.invoke(cmd, input='\n')
        expected = 'Opt []: \n\n'
        self.assertIsNone(result.exception)
        self.assertIn(expected, result.output)

    def test_prompt_dynamic_default(self):
        """Test that dynamic defaults for prompting still work."""

    def test_not_required_noninteractive(self):
        cmd = self.simple_command(required=False)
        result = self.runner.invoke(cmd, ['--non-interactive'])
        self.assertIsNone(result.exception)
        # I strip, there is typically a \n at the end
        self.assertEqual(result.output, 'None\n')

    def test_not_required_interactive(self):
        cmd = self.simple_command(required=False)
        result = self.runner.invoke(cmd, input='value\n')
        expected = 'Opt: value\nvalue\n'
        self.assertIsNone(result.exception)
        self.assertIn(expected, result.output)

    def test_not_required_noninteractive_default(self):
        cmd = self.simple_command(required=False, default='')
        result = self.runner.invoke(cmd, ['--non-interactive'])
        self.assertIsNone(result.exception)
        self.assertEqual(result.output, '\n')

    def test_not_required_interactive_default(self):
        cmd = self.simple_command(required=False, default='')
        result = self.runner.invoke(cmd, input='\nnot needed\n')
        expected = 'Opt []: \n\n'
        self.assertIsNone(result.exception)
        self.assertIn(expected, result.output)
