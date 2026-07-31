# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.block_data import BlockData
from fuser.fused_loop import FusedLoop, LoopBlock
from fuser.fused_operation import FusedOperation
from fuser.fuser_config import GlobalConfig
from fuser.pack_node import PackNode

from .packer import Packer


class MatmulPacker(Packer):
    loop: FusedLoop = LoopBlock()

    def get_headers(self) -> List[str]:
        return [
            "llk_pack.h",
            "llk_pack_matmul.h",
        ]

    def init(
        self,
        pack_node: PackNode,
        operation: FusedOperation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        buf_desc_id = pack_node.output.buf_desc_id
        subblock_r_dim = operation.block_tiles_y
        subblock_c_dim = operation.block_tiles_x
        num_subblocks_c_dim = pack_node.output.tile_count_x // subblock_c_dim
        if (
            pack_node.output.tile_count_x % subblock_c_dim != 0
            or pack_node.output.tile_count_y % subblock_r_dim != 0
        ):
            raise ValueError(
                "MatmulPacker requires block_size to tile the output tile grid "
                "exactly (tile_count_x % block_tiles_x == 0 and "
                "tile_count_y % block_tiles_y == 0)"
            )
        return f"_llk_pack_matmul_init_({buf_desc_id}, {subblock_r_dim}, {subblock_c_dim}, {num_subblocks_c_dim});\n"

    def pack(
        self,
        pack_node: PackNode,
        operation: FusedOperation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        return f"_llk_pack_matmul_(0, {block.tile_id_global});\n"
