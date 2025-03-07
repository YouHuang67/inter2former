_base_ = ['_base_/runtime.py', 'inter2former.py']
model = dict(decode_head=dict(type='DynamicLocalUpsampling'))
