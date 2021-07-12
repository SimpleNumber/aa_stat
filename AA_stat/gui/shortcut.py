from distutils import dist
import distutils.command.install as dist_install

import os
import sys
import subprocess

from .logging import get_logger


def get_install_script_dir():
     d = dist.Distribution()
     install_cmd = dist_install.install(d)
     install_cmd.finalize_options()
     return install_cmd.install_scripts


def create_shortcut():
    logger = get_logger()
    try:
        from pyshortcuts import make_shortcut
    except ImportError as e:
        logger.debug('Could not import pyshortcuts: %s', e.args[0])
        logger.debug('Trying to pip install pyshortcuts...')
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyshortcuts'], check=True)
            from pyshortcuts import make_shortcut
        except subprocess.CalledProcessError:
            logger.error('Could not install pyshortcuts.')
    try:
        make_shortcut(os.path.join(get_install_script_dir(), 'AA_stat_GUI'), name='AA_stat',
            description='AA_stat GUI', terminal=False)
    except Exception as e:
        logger.error(f'Could not create shortcut. Got a {type(e)}: {e.args[0]}')
    else:
        logger.info('Desktop shortcut created.')
