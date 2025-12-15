# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pytest
import torch
import numpy as np

# This test file is about TVM's symbolic arithmetic analysis (ConstIntBound) for Tensor Expression (TE)
# and Tensor IR (TIR) variables. PyTorch operates on concrete tensors and does not have a direct
# equivalent for symbolic bound analysis of expressions at this low-level IR stage.
# Therefore, these tests cannot be directly translated to PyTorch operations.

@pytest.mark.skip(reason="TVM's symbolic arithmetic analysis (ConstIntBound) has no direct PyTorch equivalent.")
def test_dtype_bound():
    # TODO: This test relies on tvm.arith.Analyzer and symbolic tvm.te.var with specific dtypes,
    # and the concept of ConstIntBound. PyTorch operates on concrete tensors and does not have
    # an equivalent for symbolic bound analysis.
    pass

@pytest.mark.skip(reason="TVM's symbolic arithmetic analysis (ConstIntBound) has no direct PyTorch equivalent.")
def test_cast_bound():
    # TODO: This test relies on tvm.arith.Analyzer, symbolic tvm.te.var, tvm.tir.truncmod,
    # and ConstIntBound for type casting. PyTorch does not have a direct equivalent for
    # symbolic bound analysis on IR expressions.
    pass

@pytest.mark.skip(reason="TVM's symbolic arithmetic analysis (ConstIntBound) has no direct PyTorch equivalent.")
def test_add_sub_bound():
    # TODO: This test relies on tvm.arith.Analyzer and ConstIntBound for symbolic addition/subtraction
    # of variables. PyTorch does not have a direct equivalent for symbolic bound analysis on IR expressions.
    pass

@pytest.mark.skip(reason="TVM's symbolic arithmetic analysis (ConstIntBound) has no direct PyTorch equivalent.")
def test_mul_bound():
    # TODO: This test relies on tvm.arith.Analyzer and ConstIntBound for symbolic multiplication
    # of variables. PyTorch does not have a direct equivalent for symbolic bound analysis on IR expressions.
    pass

@pytest.mark.skip(reason="TVM's symbolic arithmetic analysis (ConstIntBound) has no direct PyTorch equivalent.")
def test_truncdiv_bound():
    # TODO: This test relies on tvm.arith.Analyzer, symbolic tvm.te.var, tvm.tir.truncdiv,
    # and ConstIntBound. PyTorch does not have a direct equivalent for symbolic bound analysis on IR expressions.
    pass

@pytest.mark.skip(reason="TVM's symbolic arithmetic analysis (ConstIntBound) has no direct PyTorch equivalent.")
def test_truncmod_bound():
    # TODO: This test relies on tvm.arith.Analyzer, symbolic tvm.te.var, tvm.tir.truncmod,
    # and ConstIntBound. PyTorch does not have a direct equivalent for symbolic bound analysis on IR expressions.
    pass

@pytest.mark.skip(reason="TVM's symbolic arithmetic analysis (ConstIntBound) has no direct PyTorch equivalent.")
def test_floordiv_bound():
    # TODO: This test relies on tvm.arith.Analyzer, symbolic tvm.te.var, tvm.te.floordiv,
    # and ConstIntBound. PyTorch does not have a direct equivalent for symbolic bound analysis on IR expressions.
    pass

@pytest.mark.skip(reason="TVM's symbolic arithmetic analysis (ConstIntBound) has no direct PyTorch equivalent.")
def test_floormod_bound():
    # TODO: This test relies on tvm.arith.Analyzer, symbolic tvm.te.var, tvm.te.floormod,
    # and ConstIntBound. PyTorch does not have a direct equivalent for symbolic bound analysis on IR expressions.
    pass

@pytest.mark.skip(reason="TVM's symbolic arithmetic analysis (ConstIntBound) has no direct PyTorch equivalent.")
def test_min_max_bound():
    # TODO: This test relies on tvm.arith.Analyzer, symbolic tvm.te.var, tvm.te.min/max,
    # and ConstIntBound. PyTorch does not have a direct equivalent for symbolic bound analysis on IR expressions.
    pass

@pytest.mark.skip(reason="TVM's symbolic arithmetic analysis (ConstIntBound) has no direct PyTorch equivalent.")
def test_select_bound():
    # TODO: This test relies on tvm.arith.Analyzer, symbolic tvm.te.var, tvm.tir.Select,
    # and ConstIntBound. PyTorch does not have a direct equivalent for symbolic bound analysis on IR expressions.
    pass

@pytest.mark.skip(reason="TVM's symbolic arithmetic analysis (ConstIntBound) has no direct PyTorch equivalent.")
def test_shift_and_bound():
    # TODO: This test relies on tvm.arith.Analyzer, symbolic tvm.te.var, bitwise shifts/and,
    # and ConstIntBound. PyTorch does not have a direct equivalent for symbolic bound analysis on IR expressions.
    pass

@pytest.mark.skip(reason="TVM's symbolic arithmetic analysis (ConstIntBound) has no direct PyTorch equivalent.")
def test_mix_index_bound():
    # TODO: This test relies on tvm.arith.Analyzer, symbolic tvm.te.var, tvm.tir.truncmod, tvm.tir.truncdiv,
    # and ConstIntBound for mixed index computations. PyTorch does not have a direct equivalent for
    # symbolic bound analysis on IR expressions.
    pass

@pytest.mark.skip(reason="TVM's symbolic arithmetic analysis (ConstIntBound) has no direct PyTorch equivalent.")
def test_size_var_bound():
    # TODO: This test relies on tvm.arith.Analyzer and tvm.te.size_var for symbolic size variables,
    # and ConstIntBound. PyTorch does not have a direct equivalent for symbolic bound analysis on IR expressions.
    pass

@pytest.mark.skip(reason="TVM's symbolic arithmetic analysis (ConstIntBound) has no direct PyTorch equivalent.")
def test_let_bound():
    # TODO: This test relies on tvm.arith.Analyzer, symbolic tvm.te.var, tvm.tir.Let,
    # and ConstIntBound. PyTorch does not have a direct equivalent for symbolic bound analysis on IR expressions.
    pass

@pytest.mark.skip(reason="TVM's symbolic arithmetic analysis (ConstIntBound) has no direct PyTorch equivalent.")
def test_floormod_negative_divisor():
    # TODO: This test relies on tvm.arith.Analyzer, symbolic tvm.te.var, tvm.te.floormod,
    # and ConstIntBound with negative divisors. PyTorch does not have a direct equivalent for
    # symbolic bound analysis on IR expressions.
    pass

@pytest.mark.skip(reason="TVM's symbolic arithmetic analysis (ConstIntBound) has no direct PyTorch equivalent.")
def test_multiple_condition():
    # TODO: This test relies on tvm.arith.Analyzer, symbolic tvm.te.var, tvm.tir.all,
    # and ConstIntBound with complex conditions. PyTorch does not have a direct equivalent for
    # symbolic bound analysis on IR expressions.
    pass

if __name__ == "__main__":
    pytest.main([__file__])
