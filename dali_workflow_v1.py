# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Example 1
# Implement the default functionality of DALI's External Source
# in a single CPU process

# Command
# python3 dali_workflow_v1.py --imageFolder='./images/304A/' --num_read_processes=16 --batch_size=16

# To profile with Nsight Systems
# nsys profile -s none -t cuda,nvtx --stats=true python3 <script> <arguments>

# NOTE:
# This implementation doesn't account for duplicates

import argparse
import os
from glob import glob

import sunpy.map

import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline

from cupy import prof

# https://docs.nvidia.com/deeplearning/dali/master-user-guide/docs/examples/general/data_loading/external_input.html


class ExternalInputIterator(object):
    def __init__(self, batch_size, img_list):
        self.batch_size = batch_size
        self.files = img_list

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            fits_filename = self.files[self.i]
            f = sunpy.map.Map(fits_filename)
            batch.append(f.data)
            print(self.i, fits_filename)
            self.i = (self.i + 1) % self.n
        return (batch,)


class ExternalSourcePipeline(Pipeline):
    def __init__(self, batch_size, eii, num_threads, device_id):
        super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id, seed=12)

        self.source = ops.ExternalSource(source=eii, num_outputs=1)

    def define_graph(self):
        (fits,) = self.source()
        output = fits.gpu()  # Copy data to GPU
        return output


def run(args):
    img_list = glob(os.path.join(args.imageFolder, "*"))
    n_iter = len(img_list) // args.batch_size

    print(n_iter)

    eii = ExternalInputIterator(args.batch_size, img_list=img_list)

    pipe = ExternalSourcePipeline(batch_size=args.batch_size, eii=eii, num_threads=args.num_read_processes, device_id=0)
    pipe.build()

    for _ in range(n_iter):
        pipe_out = pipe.run()
        images = pipe_out

        # for j in range(args.batch_size):
        # 	  Do work on GPU


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imageFolder", help="Image folder location", default=[])
    parser.add_argument("--num_read_processes", help="Number of Read Processes (DALI)", default=1, type=int)
    parser.add_argument("--batch_size", help="Read batch size", default=1, type=int)

    args = parser.parse_args()

    with prof.time_range("run", 0):
        run(args)
