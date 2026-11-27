"""显式补齐当前九齿前端/SSA后端的bitcast属性传递，仅在候选构建进程启用。"""
import ast
from dataclasses import replace
import hashlib
import inspect


def enable():
    import ninetoothed.frontend.python as front
    import ninetoothed.backends.emitters.ssa as emit
    from ninetoothed.backends.emitters.triton import TritonTarget
    cls=front._ApplicationSSABuilder
    original=cls._lower_tensor_method_call
    old_cast=emit._cast_value
    record=dict(frontend_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),emitter_sha256=hashlib.sha256(inspect.getsource(old_cast).encode()).hexdigest())
    assert 'return self._lower_cast(receiver, node.args[0], operations, env)' in inspect.getsource(original)
    def lower(self,node,operations,env):
        result=original(self,node,operations,env)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='to':
            keywords={k.arg:k.value for k in node.keywords}
            if 'bitcast' in keywords:
                value=ast.literal_eval(keywords['bitcast'])
                assert isinstance(value,bool)
                if value:
                    op=operations[-1];assert op.opcode=='tensor.cast' and op.results[0].name==result.name
                    operations[-1]=replace(op,attrs={**op.attrs,'bitcast':True})
        return result
    def cast(op,value,ctx):
        if op.attrs.get('bitcast'):
            assert isinstance(ctx.target,TritonTarget), '本扩展只支持Triton后端，其他后端不得静默使用数值转换'
            dtype=emit._resolved_cast_dtype(op,ctx)
            assert dtype in ('uint32','float32'),dtype
            return f'{value}.to(tl.{dtype}, bitcast=True)'
        return old_cast(op,value,ctx)
    cls._lower_tensor_method_call=lower;emit._cast_value=cast
    return record
