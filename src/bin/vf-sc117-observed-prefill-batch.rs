//! Compare a real two-position prefill expert batch with ungrouped native execution.
use std::{collections::BTreeMap,fs,time::{Duration,SystemTime,UNIX_EPOCH}};
use voltafabric::{moe::{Choice,Registry,Replica,Stamp},native_ggml_cpu::{Binding,Resident},native_ledger::NativeBatchLedger,routing::RoutingProblem,session::{Endpoint,Session},sha256,strata_native_layout::NativeLayout};
type E=Box<dyn std::error::Error>;
const INDEX:&str="adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";
const WORKER:&str="ce6786e7d35182baa64c204ff8de73c190923baa23c560e6c40df497ab079f32";
const SPEC:&str="3c20888b55920f5b108115ca4d24208de578755bc286c7c8ae8301959ecfea20";
const SIZE:u64=50_292_326_400;
fn run()->Result<serde_json::Value,E>{
 let a:Vec<_>=std::env::args().collect(); if a.len()!=7{return Err("usage: program SPEC PACK INDEX LEDGER WORKER PLAN_SHA".into());}
 let raw=fs::read(&a[1])?; if sha256::hex(&raw)!=SPEC{return Err("spec mismatch".into())} let spec:serde_json::Value=serde_json::from_slice(&raw)?;
 let index=fs::read_to_string(&a[3])?; if sha256::hex(index.as_bytes())!=INDEX || sha256::hex(&fs::read(&a[5])?)!=WORKER || !sha256::is_hex_digest(&a[6]) {return Err("identity mismatch".into())}
 let records=spec["inputs"].as_array().ok_or("records")?; if records.len()!=2{return Err("records count".into())}
 let mut inputs=Vec::new();let mut routes=Vec::new();
 for r in records {let bits=r["activation_bits"].as_array().ok_or("activation")?; if bits.len()!=2560{return Err("activation length".into())}
  let x:Vec<f32>=bits.iter().map(|b|u32::try_from(b.as_u64().unwrap_or(u64::MAX)).map(f32::from_bits)).collect::<Result<_,_>>()?;
  if x.iter().any(|v|!v.is_finite()) || sha256::hex(&x.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())!=r["activation_sha256"].as_str().ok_or("hash")? {return Err("activation mismatch".into())}
  let cs=r["choices"].as_array().ok_or("choices")?; if cs.len()!=10{return Err("choice count".into())}
  let mut choices=Vec::new();for c in cs{let expert=u32::try_from(c["expert"].as_u64().ok_or("expert")?)?;let weight=c["weight"].as_f64().ok_or("weight")? as f32; if !weight.is_finite() || weight<0.0{return Err("weight".into())} choices.push(Choice{expert,weight:f64::from(weight)});} inputs.push(x);routes.push(choices);
 }
 let meta=spec["experts"].as_object().ok_or("experts")?;if meta.len()!=17{return Err("expert count".into())}
 let layout=NativeLayout::parse_sc117_v3(&index,SIZE)?;let mut pack=fs::File::open(&a[2])?;let mut copies=Vec::new();let mut buffers=Vec::new();let mut references:BTreeMap<(usize,u32),Vec<u32>>=BTreeMap::new();let mut counters=BTreeMap::from([(2,0u32),(3,0u32)]);
 for (key,value) in meta {let expert:u32=key.parse()?;let rank=u32::try_from(value["rank"].as_u64().ok_or("rank")?)?;let local=counters.get_mut(&rank).ok_or("rank outside local plan")?;let binding=Binding{model_revision:"sc117-local-observation".into(),rank,layer:3,expert,local:*local,epoch:1,index_sha256:INDEX.into(),blob_sha256:value["sha256"].as_str().ok_or("digest")?.into()};*local+=1;
  let blob=layout.read_verified_expert(&mut pack,3,expert.into(),INDEX,&binding.blob_sha256)?;
  let mut resident=Resident::load(&layout,blob,binding.clone())?;
  for (t,route) in routes.iter().enumerate(){if route.iter().any(|c|c.expert==expert){references.insert((t,expert),resident.forward(&inputs[t])?.iter().map(|v|v.to_bits()).collect());}}
  drop(resident);let blob=layout.read_verified_expert(&mut pack,3,expert.into(),INDEX,&binding.blob_sha256)?;copies.push(Replica{expert,rank,local:binding.local});buffers.push((binding,blob));
 }
 let timeout=Duration::from_secs(60);let ledger=NativeBatchLedger::open(std::path::Path::new(&a[4]))?;let mut sessions=Vec::new();
 for rank in [2,3]{let mut s=Session::connect(&Endpoint{rank,argv:vec![a[5].clone(),"--guarded-local".into(),"--expected-exe-sha256".into(),WORKER.into()]},timeout)?;for (b,blob) in &buffers{if b.rank==rank{s.load_native_ggml(b,&index,SIZE,&blob.bytes,timeout)?;}}s.seal_native_ggml(counters[&rank] as usize,timeout)?;sessions.push(s);}
 let order=BTreeMap::from([(2,(0..counters[&2]).collect()),(3,(0..counters[&3]).collect())]);
 let mut problem=RoutingProblem{hidden:2560,experts:copies.iter().map(|r|r.expert).collect(),routes,registry:Registry{epoch:1,copies},stamp:Stamp{run_id:format!("r554-{}-{}",std::process::id(),SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()),model_revision:"sc117-local-observation".into(),plan_id:a[6].clone(),layer:3,batch:0,epoch:1},preferred_owners:BTreeMap::new(),resident_order:order};
 let out=Session::exec_native_ggml_unweighted_f32(&mut sessions,&problem,&inputs,&ledger,timeout)?;problem.stamp.batch=1;let reduced=Session::exec_native_ggml_batch(&mut sessions,&problem,&inputs,&ledger,timeout)?;
 for (t,rows) in out.iter().enumerate(){if rows.len()!=10{return Err("rows".into())}for (s,row) in rows.iter().enumerate(){let actual:Vec<u32>=row.iter().map(|v|(*v as f32).to_bits()).collect();if references.get(&(t,problem.routes[t][s].expert))!=Some(&actual){return Err("ungrouped reference mismatch".into())}}}
 let mut receipts=Vec::new();for s in sessions{let raw=s.traffic.native_ggml_raw_execs;let r=s.close(timeout);if raw!=2||r.bye_execs!=Some(2)||r.exit_code!=Some(0)||r.error.is_some()||r.killed{return Err("worker close".into())}receipts.push(serde_json::json!({"rank":r.rank,"pid":r.pid,"raw_execs":raw,"exit_code":r.exit_code}));}
 Ok(serde_json::json!({"run_id":problem.stamp.run_id,"plan_sha256":a[6],"spec_sha256":SPEC,"raw_f32_bits":out.iter().map(|t|t.iter().map(|r|r.iter().map(|v|(*v as f32).to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>()).collect::<Vec<_>>(),"collected_f64_bits":reduced.iter().map(|r|r.iter().map(|v|v.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>(),"receipts":receipts}))
}
fn main(){match run(){Ok(v)=>println!("{v}"),Err(e)=>{eprintln!("observed batch refused: {e}");std::process::exit(1)}}}
