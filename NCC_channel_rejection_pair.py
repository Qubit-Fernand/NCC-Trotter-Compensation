"""follow pseudocode"""

import argparse
import math
from functools import lru_cache

import numpy as np
from scipy.linalg import expm
from tqdm import tqdm

from NCC_channel_normal import (
    apply_channel_term,
    build_static_data as build_channel_static_data,
    build_tilde_V as build_channel_tilde_V,
    trace_norm,
)
from Pauli_Hamiltonian_BCH import build_periodic_ab, cached_pauli_matrix_from_label, phi_term, tilde_F_term, commutator

from NCC_channel_rejection import (
    build_static_data as build_rejection_static_data,
    build_tilde_V as build_rejection_tilde_V,
    sample_term,
)
