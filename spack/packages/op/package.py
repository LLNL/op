# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# ----------------------------------------------------------------------------
# If you submit this package back to Spack as a pull request,
# please first remove this boilerplate and all FIXME comments.
#
# This is a template package file for Spack.  We've put "FIXME"
# next to all the things you'll want to change. Once you've handled
# them, you can save this file and test your package like this:
#
#     spack install op
#
# You can edit this file again by typing:
#
#     spack edit op
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack import *

import os
import socket
from os.path import join as pjoin

def get_spec_path(spec, package_name, path_replacements={}, use_bin=False):
    """Extracts the prefix path for the given spack package
       path_replacements is a dictionary with string replacements for the path.
    """

    if not use_bin:
        path = spec[package_name].prefix
    else:
        path = spec[package_name].prefix.bin

    path = os.path.realpath(path)

    for key in path_replacements:
        path = path.replace(key, path_replacements[key])

    return path

class Op(CMakePackage):
    """FIXME: Put a proper description of your package here."""

    # FIXME: Add a proper url for your package's homepage here.
    homepage = "https://lc.llnl.gov/gitlab/wong125/op"
    git      = "ssh://git@czgitlab.llnl.gov:7999/wong125/op.git"

    version('master', branch='master', submodules=True, preferred=True)
    
    # -----------------------------------------------------------------------
    # Variants
    # -----------------------------------------------------------------------
    
    variant('debug', default=False,
            description='Enable runtime safety and debug checks')

    depends_on('nlopt@2.7.0~python')
    depends_on("mpi")
    depends_on("cmake@3.14:")


    def cmake_args(self):
        spec = self.spec

        path_replacements = {}
        
        nlopt_dir = get_spec_path(spec, "nlopt", path_replacements)
        
        args = [
            self.define('NLOPT_DIR', nlopt_dir),
        ]

        return args

