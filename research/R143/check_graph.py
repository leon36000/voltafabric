import collections,email,hashlib,json
from pathlib import Path
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import parse_tag
from packaging.utils import canonicalize_name,parse_wheel_filename
from packaging.version import Version
p=Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/r143-pep658-metadata'); r=json.loads((p/'terminal.json').read_text());envtag=json.loads((p/'pc1-environment-tags.json').read_text());env=envtag['environment'];tags=set()
for s in envtag['tags']: tags.update(parse_tag(s))
selected_path=Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/gitlab-import-r113-20261028/repository/research/R141/parsed-candidates.json');selected=json.loads(selected_path.read_text());table={};bad=[]
for c in selected:
 name=canonicalize_name(c['name']);version=Version(c['version'])
 if name in table:bad.append({'type':'duplicate','name':name});continue
 table[name]={'version':version,'metadata':None,'filename':c['link'].split('/')[-1]}
 try:
  nm,v,_,wt=parse_wheel_filename(table[name]['filename']);
  if canonicalize_name(nm)!=name or v!=version or not(wt&tags):bad.append({'type':'tag_or_identity','name':name})
 except Exception as e:bad.append({'type':'invalid_filename','name':name,'error':type(e).__name__})
for x in r['candidates']:
 if x['status']!='VERIFIED':continue
 n=canonicalize_name(x['name']);key=hashlib.sha256((x['name']+'=='+x['version']).encode()).hexdigest();data=(p/'metadata'/(key+'.metadata')).read_bytes()
 if hashlib.sha256(data).hexdigest()!=x['metadata_sha256']:bad.append({'type':'metadata_hash','name':n});continue
 table[n]['metadata']=email.message_from_bytes(data)
fallback=json.loads((p/'fallback-receipt.json').read_text()); raw=(p/'pydantic-core.metadata').read_bytes(); assert fallback['status']=='VERIFIED_WHEEL_AND_METADATA' and hashlib.sha256(raw).hexdigest()==fallback['metadata_sha256']; table['pydantic-core']['metadata']=email.message_from_bytes(raw)
root=canonicalize_name('1cat-vllm');m=(p/'1cat.metadata').read_bytes();assert hashlib.sha256(m).hexdigest()=='8c21ab87c8454d229c175deb0b8927f55529bc40f468113fb1f015f0001b0cd9';table[root]['metadata']=email.message_from_bytes(m)
queue=collections.deque([(root,'')]);seen=set();edges=[];missing_meta=set();unknown_extras=[]
while queue:
 name,extra=queue.popleft()
 if (name,extra) in seen:continue
 seen.add((name,extra));item=table.get(name)
 if not item:bad.append({'type':'missing_package','name':name});continue
 meta=item['metadata']
 if meta is None:missing_meta.add(name);continue
 if extra and extra not in {canonicalize_name(x) for x in meta.get_all('Provides-Extra',[])}:unknown_extras.append({'name':name,'extra':extra})
 rp=meta.get('Requires-Python')
 if rp:
  from packaging.specifiers import SpecifierSet
  if Version(env['python_full_version']) not in SpecifierSet(rp):bad.append({'type':'python_requirement','name':name,'requirement':rp})
 for raw in meta.get_all('Requires-Dist',[]):
  try:req=Requirement(raw)
  except Exception as e:bad.append({'type':'invalid_requirement','name':name,'requirement':raw});continue
  marker_env=dict(env,extra=extra)
  if req.marker is not None and not req.marker.evaluate(marker_env):continue
  target=canonicalize_name(req.name);edges.append({'from':name,'extra':extra,'requirement':raw,'to':target});candidate=table.get(target)
  if req.url:bad.append({'type':'direct_url_unverified','from':name,'to':target});continue
  if not candidate:bad.append({'type':'missing_package','from':name,'to':target,'requirement':raw});continue
  if candidate['version'] not in req.specifier:bad.append({'type':'version_conflict','from':name,'to':target,'requirement':raw,'selected':str(candidate['version'])})
  queue.append((target,''))
  for e in req.extras:queue.append((target,canonicalize_name(e)))
out={'status':'GRAPH_INCOMPLETE' if missing_meta or bad or unknown_extras else 'GRAPH_VERIFIED_FOR_PC1_ONLY','selected_count':len(table),'verified_wheel_metadata':sum(x['metadata'] is not None for x in table.values()),'pc1_environment':env,'visited_package_extra_states':len(seen),'active_edges':len(edges),'unvisited_selected':sorted(set(table)-{x[0] for x in seen}),'missing_metadata':sorted(missing_meta),'violations':bad,'unrecognized_extras':unknown_extras,'limits':'PC1 marker and tag environment only; unverified package metadata and missing package transitive edges remain unknown; only pydantic-core wheel bytes downloaded; other remote wheel bytes not downloaded; 1Cat donor override 0.1.10 not applied.'}
(p/'graph.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('status','selected_count','verified_wheel_metadata','visited_package_extra_states','active_edges','unvisited_selected','missing_metadata','violations','unrecognized_extras')},sort_keys=True))
