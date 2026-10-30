"""Research admission for 1Cat's global lazy per-expert weight filter.

READY bindings must be supplied by a trusted live-session authority; dictionaries
alone are not proof.  This function never creates or mutates an engine loader.
"""
import re
class AdmissionError(ValueError): pass

def filter_from_bindings(*, model_revision, plan_id, epoch, layers, expert_count,
                         remote_ids, bindings, load_format, strategy, multithread,
                         prefetch, eplb):
    if (not isinstance(model_revision,str) or not model_revision or not isinstance(plan_id,str) or not plan_id
        or type(epoch)!=int or epoch<0 or type(expert_count)!=int or not 0<expert_count<=4096
        or not isinstance(layers,(tuple,list)) or not layers or any(type(x)!=int or x<0 for x in layers)
        or len(set(layers))!=len(layers)):
        raise AdmissionError('invalid model, plan, epoch, layers or domain')
    if load_format!='safetensors' or strategy!='lazy' or multithread is not False or prefetch is not False or eplb is not False:
        raise AdmissionError('unsupported loader mode or placement policy')
    if not isinstance(remote_ids,(set,frozenset)) or any(type(x)!=int or not 0<=x<expert_count for x in remote_ids):
        raise AdmissionError('invalid remote expert IDs')
    if not isinstance(bindings,(tuple,list)):
        raise AdmissionError('invalid binding evidence')
    ready=set();seen=set()
    for b in bindings:
        required={'model_revision','plan_id','epoch','layer','expert','rank','artifact_sha256','state'}
        if type(b)!=dict or set(b)!=required or b['model_revision']!=model_revision or b['plan_id']!=plan_id or type(b['epoch'])!=int or b['epoch']!=epoch:
            raise AdmissionError('binding identity differs')
        if (type(b['layer'])!=int or b['layer'] not in layers or type(b['expert'])!=int or not 0<=b['expert']<expert_count
            or type(b['rank'])!=int or b['rank']<0 or b['state']!='READY'
            or not isinstance(b['artifact_sha256'],str) or re.fullmatch('[0-9a-f]{64}',b['artifact_sha256']) is None):
            raise AdmissionError('invalid or unavailable owner binding')
        key=(b['layer'],b['expert'],b['rank'])
        if key in seen:raise AdmissionError('duplicate owner binding')
        seen.add(key);ready.add((b['layer'],b['expert']))
    if any((layer,expert) not in ready for layer in layers for expert in remote_ids):
        raise AdmissionError('remote expert is not ready in every layer')
    # 1Cat's single set applies globally to all layers. Unlisted experts remain local.
    return set(range(expert_count))-remote_ids
