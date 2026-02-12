/*
 *  Original file: hartree-fock++.cc
 *  Copyright (C) 2004-2026 Edward F. Valeev
 *
 *  This file is part of Libint library.
 *
 *  Libint library is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Libint library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with Libint library.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "hartree_fock.h"

// Global variable definitions
shellpair_list_t obs_shellpair_list;  // shellpair list for OBS
shellpair_data_t obs_shellpair_data;  // shellpair data for OBS

// libint2 namespace variable definition
namespace libint2 {
int nthreads;
}  // namespace libint2
