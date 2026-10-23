'use strict';
(async()=>{
 const r=await fetch('results-20261019.json');if(!r.ok)throw Error('Evidence could not be loaded');const d=await r.json();
 const esc=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const num=(n,p=3)=>Number.isFinite(n)?n.toFixed(p):'—';
 const range=a=>a.map(n=>num(n,4)).join('–');
 const row=(id,cells)=>{const e=document.getElementById(id);if(e)e.insertAdjacentHTML('beforeend','<tr>'+cells.map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>')};
 const labels={rack:'179-chain rack · 92,694 bodies',ladder50:'Small control · 200 bodies',yharness:'Single Y · 1,197 bodies',yfarm72:'72 mixed Ys · 17,802 bodies'};
 for(const s of d.scale){if(s.scene==='yfarm72')continue;row('scale-rows',[labels[s.scene]+' · '+s.device,range(s.baseline_range_ms),range(s.candidate_range_ms),num(s.mean_speedup,3)+'×',s.device==='cpu'?'No reliable general benefit':'Saved states bitwise · pre-hardening']);}
 row('scale-rows',['72 mixed Ys · final source',range(d.final_scale.baseline_range_ms),range(d.final_scale.candidate_range_ms),num(d.final_scale.mean_ratio,3)+'×','States differ · throughput only']);
 for(const s of d.scale)for(const [name,r]of Object.entries(s.runs))addSetup(labels[s.scene]+' '+s.device+' '+name,r.setup);
 for(const [name,r]of Object.entries(d.final_scale.runs))addSetup('Final mixed-Y '+name,r.setup);
 for(const r of d.contact)addSetup(r.prefix,r.setup);
 function addSetup(name,s){row('setup-rows',[name,num(s.model_build_s),num(s.constructor_s??s.solver_constructor_s),num(s.first_two_steps_compile_and_simulation_s),num(s.capture_s),num(s.diagnostics_s)]);}
 for(const t of d.friction)row('toy-rows',[t.mu+' / G'+t.g,...t.measurements.hold_rod_rms_um.map(v=>num(v,5)),num(t.measurements.mean_step_us_excluding_first_frame/1000,4),t.status]);
 for(const target of document.querySelectorAll('[data-pair]')){
  for(const m of d.media.filter(m=>m.group===target.dataset.pair)){
   const figure=document.createElement('figure');const v=document.createElement('video');v.controls=true;v.muted=true;v.playsInline=true;v.preload='none';v.poster=m.poster;v.src=m.src;v.dataset.seconds=m.encoded_duration_s;v.dataset.sha256=m.output_sha256;v.setAttribute('aria-label',m.title);figure.append(v);
   const caption=document.createElement('figcaption');caption.innerHTML='<strong>'+esc(m.title)+'</strong><p class="caption">'+esc(m.caption)+'</p><p class="caption">'+num(m.encoded_duration_s,3)+' s encoded · '+m.fps+' fps · '+m.playback_speed+'× playback · Newton ViewerGL</p>';figure.append(caption);target.append(figure);
  }
 }
 for(const m of d.media)row('render-rows',[m.title,num(m.encoded_duration_s),num(m.render_cost_s),m.playback_speed+'×',m.output_sha256]);
 document.body.dataset.loaded='true';window.reportEvidence=d;
})().catch(e=>{const p=document.createElement('p');p.className='error';p.textContent=String(e);document.querySelector('main').prepend(p);console.error(e);});
