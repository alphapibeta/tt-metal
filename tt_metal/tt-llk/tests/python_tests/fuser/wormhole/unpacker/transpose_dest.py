# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fused_operation import FusedOperation
from fuser.fuser_config import GlobalConfig

from .unpack_a import UnpackerA


class TransposeDestUnpacker(UnpackerA):
    def unpack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_llk_unpack_set_srcb_dummy_valid_();\n"

    def perf_set_valid(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_llk_unpack_set_srcb_dummy_valid_();\n"

    def perf_clear_valid(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
