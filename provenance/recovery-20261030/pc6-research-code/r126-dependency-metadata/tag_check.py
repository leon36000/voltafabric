import json, platform
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.version import Version
from importlib.metadata import requires
files=['tilelang-0.1.9-cp38-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl','tilelang-0.1.10-cp38-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl','apache_tvm_ffi-0.1.9-cp312-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl','apache_tvm_ffi-0.1.10-cp312-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl','torch-2.10.0-cp312-cp312-manylinux_2_28_x86_64.whl']
tags=set(sys_tags());reqs=[Requirement(x) for x in requires('1cat-vllm')]; print(json.dumps({'platform':platform.platform(),'wheel_tag_compatible':{n:bool(tags & parse_wheel_filename(n)[3]) for n in files},'declared_specifiers':{r.name:{v:Version(v) in r.specifier for v in ['0.1.9','0.1.10']} for r in reqs if r.name.lower().replace('_','-') in ('tilelang','apache-tvm-ffi')},'scope':'metadata and wheel tags only; no wheels installed, imports or GPU'}))
